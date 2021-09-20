# Authors: Tatiana Engel <engel@cshl.edu>,
#          Mikhail Genkin <mgenkin@cshl.edu>
#

"""This is a part of neuralflow package/EnergyModel class.
This source file contains the definition of energy model class."""


import numpy as np

from sklearn.base import BaseEstimator
from . import firing_rate_models, peq_models, PDE_Solve

# Import source files:
from .energy_model_llcalc import _get_loglik_data, _get_loglik_seq, _get_EV_solution
from .energy_model_data_generation import generate_data, \
    _generate_data, transform_spikes_to_isi, _generate_inhom_poisson, \
    _generate_diffusion, _sample_from_p
from .energy_model_fit_and_predict import score, fit, calc_F, calc_peq, SaveResults
from .energy_model_GD import _GD_optimization, FeatureComplexity, FeatureComplexityFderiv
from .energy_model_error_handling import check_em_parameters, _check_data, _check_optimization_options, _check_sequence


class EnergyModel(BaseEstimator):
    """Energy Model class.


    Parameters
    ----------
    num_neuron : int
        A number of neuronal responses. The default is 1.
    firing_model : list
        For each neuron, this list contains the firing rate function. Each entry is either a function that returns an array of firing rate values,
        or a dictionary that specifies a model from ``firing_rate_models.py`` file.
        The default is [{"model": "linear", "params": {"r_slope": 50, "r_bias": 60}}].
    peq_model : dictionary or callable
        Equilibrium probability distribution density (which defines the driving force and Langevin potential, see Supplementary Section 1.1 in Genkin et. al. 2020 paper).
        This quantitiy defines the Potential function via Phi(x)=-log(peq(x)).
        Can be specified as a function, or as a dictionary that specifies one of the models from ``peq_models.py`` file.
        The default is {"model": "linear_pot", "params": {"slope": -2.65}}.
    p0_model : dictionary or callable or None
        Initial probability distribution. Can be specified as a function, or as a dictionary that specifies one of the models from
        ``peq_models.py`` file. If set to None, it will be assumed to be equal to peq (the equilibirum/stationary inference).
        The default is None.
    D0 : float
        Diffusion, or noise magnitude value. The default is 0.56.
    boundary_mode : string
        Specify boundary mode that will set boundary conditions for the optimization/data generation.
        Possbile options are "reflecting", "absorbing", None. If set to None, the boundary conditions can be specified
        in pde_solve_param dictionary. The default is None.
    Nv : int
        A number of retained eigenvalues/eigenvectors of the operator H. Too low value can degrade the quality, while too high value can
        result in long computational times. The default value is N-2, where N is the total number of grid points(for SEM N=Ne*(Np-1)+1).
        Since there are two boundary conditions, this is the maximum possible number of the retained eigenvalues. Note, that for numberical stability it is preferable to use
        the default value due to possible parasitic oscillation that arise from absorption operator. This is because for the operator exp(-Hdt) the
        contributions from the large eigenvalues decays exponentially, while for the absorption operator A=H0 it decays linearly, and generally cannot be neglected.
    pde_solve_param : dictionary
        Dictionary of the parameters for PDE_Solve class, see also docstrings in ``PDE_Solve.py`` file.
        Optional, as every entry has the default value.
        Possible entries:

        xbegin : float
            The left boundary of the latent state. The default is -1.
        xend : float
            The right boundary of the latent state. The default is 1.
        method : dictionary
            A dictionary that contains 2 keys:
                name : string
                    Specifies the method for the numerical solution of EV problem, can be either 'FD' or 'SEM' (forward differences or spectral elements method).
                gridsize : dictionary
                    Specifies number of grid size points N for 'FD' method, or Np and Ne for 'SEM' method (Ne is the number of elements, Np is the number of grid points in each element).
            The default is {'name': 'SEM', 'gridsize': {'Np': 8, 'Ne': 256}}.
        BoundCond : dictionary
            A dictionary that specifies boundary conditions (Dirichlet, Neumann or Robin).
            If boundary mode is supplied, it will overwrite this variable. For this input pararmeter to be in effect, the boundary_mode
            should be set to None. The default is {'leftB': 'Neumann', 'rightB': 'Neumann'}.
        grid_mode_calc : str
            Specify how to calculate SEM grid collocation points.
            Availiable options:
            'built_in': using numpy.polynomial module
            'newton': using Newton's method to calculate zeros of Legendre polinomial for the GLL grid
            The default is 'newton'.
        BC_method : str
            Specify the method of boundary condition handling when transforming the EV problem into linear system of equations.
            Availiable options:
            'projection': use projection operator.
            'bound_subst': use boundary condition substitution into the first and the last equations of the associated linear system.
            The default is 'projection'
        int_mode : str
            Specify the integration mode.
            Availiable options:
                'full' - use full integration matrix.
                'sparse' - use sparse integration matrix with a bias vector.
            The default is 'full'. See Supplementary Information for M. Genkin, T. A. Engel, Nat Mach Intell 2, 674–683 (2020) for details.

    verbose : bool
        A boolean specifying the verbose level. The true value displays more messages. The default is False.


    Attributes
    ----------
    x_d_ : numpy array, dtype=float
        Grid points. Will be calculated with ``PDE_Solve`` class.

    w_d_ : numpy array, dtype=float
        The corresponding SEM weights (only for SEM method).

    dmat_d_ : numpy array, dtype=float
        The matrix for numerical differentiation.

    peq_ : numpy array, dtype=float
        The equilibrium probability density evaluated on SEM grid
    p0_ : numpy array, dtype=float
        The initial distribution p0.
    D_ : float
        Noise intensity.
    fr_: numpy array, dtype=float
        An array with firing rate functions evaluated on SEM grid. The number of columns is equal to the number of neuronal responses.
    N : int
        Number of grid points
    Np : int
        Number of grid points per element (only SEM method).
    Ne : int
        Number of elements (only SEM method).
    """

    _firing_model_types = {
        'rectified_linear': firing_rate_models.rectified_linear,
        'linear': firing_rate_models.linear,
        'peaks': firing_rate_models.peaks,
        'sinus': firing_rate_models.sinus,
        'cos_square': firing_rate_models.cos_square,
        'custom': firing_rate_models.custom
    }

    _peq_model_types = {
        'cos_square': peq_models.cos_square,
        'cos_fourth_power': peq_models.cos_fourth_power,
        'single_well': peq_models.single_well,
        'double_well': peq_models.double_well,
        'uniform': peq_models.uniform,
        'mixture': peq_models.mixture,
        'custom': peq_models.custom,
        'jump_spline2': peq_models.jump_spline2,
        'linear_pot': peq_models.linear_pot,
    }

    _boundary_modes = ['reflecting', 'absorbing']
    _optimizer_types = ['GD']
    _parameters_to_optimize = ['F', 'F0', 'D']

    # Import class functions from different source files
    _get_loglik_data = _get_loglik_data
    _get_loglik_seq = _get_loglik_seq
    _get_EV_solution = _get_EV_solution
    generate_data = generate_data
    _generate_data = _generate_data
    transform_spikes_to_isi = transform_spikes_to_isi
    _generate_inhom_poisson = _generate_inhom_poisson
    _generate_diffusion = _generate_diffusion
    _sample_from_p = _sample_from_p
    _GD_optimization = _GD_optimization
    FeatureComplexity = FeatureComplexity
    FeatureComplexityFderiv = FeatureComplexityFderiv
    score = score
    fit = fit
    SaveResults = SaveResults
    calc_peq = calc_peq
    calc_F = calc_F
    _check_data = _check_data
    _check_sequence = _check_sequence
    _check_optimization_options = _check_optimization_options

    @check_em_parameters
    def __init__(self, num_neuron, firing_model, peq_model, p0_model, boundary_mode, D0, Nv, pde_solve_param, verbose):

        # Enforce the boundary conditions for the chosen boundary mode
        if boundary_mode is not None:
            bcs = {"reflecting": "Neumann",
                   "absorbing": "Dirichlet"}[boundary_mode]
            if verbose and "BoundCond" in pde_solve_param:
                if pde_solve_param["BoundCond"] != {'leftB': bcs, "rightB": bcs}:
                    print('WARNING: Enforcing {} boundary conditions'.format(bcs))
            pde_solve_param["BoundCond"] = {'leftB': bcs, "rightB": bcs}
        self.boundary_mode = boundary_mode

        # create an instance of the PDESolve class for numerical solution of PDEs for ll calculation
        self.pde_solve_ = PDE_Solve.PDESolve(**pde_solve_param)

        # Initiliaze the parameters
        self.num_neuron = num_neuron
        self.firing_model = firing_model
        self.peq_model = peq_model
        self.p0_model = p0_model
        self.boundary_mode = boundary_mode
        self.D0 = D0
        self.Nv = Nv if Nv is not None else self.pde_solve_.N - 2
        self.pde_solve_param = pde_solve_param
        self.verbose = verbose
        self.self_var = None

        # copy the pointers for the grid points, integration weights, derivative matrix, number of grid points,
        # and Integration function for convinience
        self.x_d_ = self.pde_solve_.x_d
        self.w_d_ = self.pde_solve_.w_d
        self.dmat_d_ = self.pde_solve_.dmat_d
        self.N = self.pde_solve_.N
        self.Integrate = self.pde_solve_.Integrate

        # initialize peq
        if len(self.peq_model_.__code__.co_varnames)==2:
            self.peq_ = self.peq_model_(self.x_d_, self.w_d_)
        else:
            self.peq_ = self.peq_model_(self.x_d_)
        self.peq_/=np.sum(self.peq_*self.w_d_)
        
        # initialize p0
        if self.p0_model is not None:
            if len(self.p0_model_.__code__.co_varnames)==2:
                self.p0_ = self.p0_model_(self.x_d_, self.w_d_)
            else:
                self.p0_ = self.p0_model_(self.x_d_)
            self.p0_/=np.sum(self.p0_*self.w_d_)
        else:
            self.p0_ = None
        
        # initialize D
        self.D_ = self.D0
        # initialize Firing Rates
        num_neuron = len(self.firing_model_)
        if self.num_neuron != num_neuron:
            print(
                f'Warining: {num_neuron} neurons was found, while the parameter num_neuron was set to {self.num_neuron}! Setting num_neuron to the correct value.')
            self.num_neuron = num_neuron
        self.fr_ = np.empty((self.N, self.num_neuron), dtype='float64')
        for i in range(self.num_neuron):
            self.fr_[:, i] = np.maximum(self.firing_model_[i](self.x_d_),0)
        self.converged_ = False
