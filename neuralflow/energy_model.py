# Authors: Tatiana Engel <engel@cshl.edu>,
#          Mikhail Genkin <mgenkin@cshl.edu>         
#


import numpy as np
import numbers

from sklearn.base import BaseEstimator
from . import firing_rate_models, peq_models, PDE_Solve

#Import source files:
from .energy_model_llcalc import _get_loglik_data, _get_loglik_seq, _get_EV_solution
from .energy_model_data_generation import generate_data, \
    _generate_data, transform_spikes_to_isi, _generate_inhom_poisson, \
    _generate_diffusion, _sample_from_p
from .energy_model_fit_and_predict import score, fit, calc_F, calc_peq, SaveResults
from .energy_model_GD import _GD_optimization, FeatureComplexity, FeatureComplexityFderiv




class EnergyModel(BaseEstimator):
    """Energy Model class.


    Parameters
    ----------
    num_neuron : int 
        A number of neuronal responses.    
    firing_model : list
        For each neuron, this list contains the firing rate function. Each entry is either a function that returns an array of firing rate values, 
        or a dictionary that specifies a model from ``firing_rate_models.py`` file. 
        The default is [{"model": "rectified_linear", "params": {"r_slope": 100, "x_thresh": -1}}].    
    peq_model : string or callable
        Equilibrium probability distribution density (which specifies driving force or potential, see Supplementary Section 1.1).
        Can be specified as a function, or as a dictionary that specifies one of the models from ``peq_models.py`` file. 
        The default is {"model": "double_well", "params": {"xmin": 0.6, "xmax": 0.0, "depth": 2}}.
    D0 : float
        Diffusion, or noise magnitude value. The default is 10.
    Nv : int
        A number of retained eigenvalues/eigenvectors of the operator H. Too low value can degrade resolution, while too high value can 
        increase the computational time. The default value is 64. 
    pde_solve_param : dictionary 
        Dictionary of the parameters for PDE_Solve class, see also ``PDE_Solve.py`` file. 
        Optional, as every entry has the default value.
        Possible entries:
            
        xbegin : float
            The left boundary of the latent state. The default is -1.
        xend : float
            The right boundary of the latent state. The default is 1.
        method : dictionary
            A dictionary that contains 2 keys: 'name' and 'gridsize'.
            'name' specifies the method for the numerical solution of EV problem, can be either 'FD' or 'SEM' 
            (forward differences or spectral elements method). 'gridsize' is a dictionary that contains number of grid size points N for 'FD' method, 
            or Np and Ne for 'SEM' method (Ne is the number of elements, Np is the number of grid points in each element).
            The default is {'name': 'SEM', 'gridsize': {'Np': 8, 'Ne': 256}}.
        BoundCond : dictionary
            A dictionary that specifies boundary conditions (Dirichlet, Neumann or Robin).
            The default is {'leftB': 'Neumann', 'rightB': 'Neumann'}
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
                'sparse' - use sparse integration matrix with bias.
            The default is 'full'. See Supplementary Materials 2.3 for details.
            
    verbose : bool
        A boolean specifying the verbose level. The true value enables displaying more messages. The default is False.


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
    D_ : float
        Noise intensity.
    fr_: numpy array, dtype=float
        An array with firing rate functions evaluated on SEM grid. The number of columns is equal to the number of neuronal responses.

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
        'custom': peq_models.custom
    }

    _optimizer_types = ['GD']

    #Import class functions from different source files
    _get_loglik_data=_get_loglik_data
    _get_loglik_seq=_get_loglik_seq
    _get_EV_solution=_get_EV_solution
    generate_data=generate_data
    _generate_data=_generate_data
    transform_spikes_to_isi=transform_spikes_to_isi
    _generate_inhom_poisson=_generate_inhom_poisson
    _generate_diffusion=_generate_diffusion
    _sample_from_p=_sample_from_p
    _GD_optimization=_GD_optimization
    FeatureComplexity=FeatureComplexity
    FeatureComplexityFderiv=FeatureComplexityFderiv
    score=score
    fit=fit
    SaveResults=SaveResults
    calc_peq=calc_peq
    calc_F=calc_F    

    def __init__(self, num_neuron=1,
                 firing_model=[{"model": "rectified_linear", "params": {"r_slope": 100, "x_thresh": -1}}],
                 peq_model={"model": "double_well", "params": {"xmin": 0.6, "xmax": 0.0, "depth": 2}},
                 D0=10, Nv=64, pde_solve_param={}, verbose=False):
        
        # create instance of the PDESolve class for numerical solution of PDEs for ll calculation
        self.pde_solve_ = PDE_Solve.PDESolve(**pde_solve_param)
        
        #Initiliaze the parameters
        self.num_neuron = num_neuron
        self.firing_model = firing_model
        self.peq_model = peq_model
        self.D0 = D0
        self.Nv = Nv
        self.verbose = verbose
        
        # copy here grid points, integration weights and derivative matrix for convinience of use
        self.x_d_ = self.pde_solve_.x_d
        self.w_d_ = self.pde_solve_.w_d
        self.dmat_d_ = self.pde_solve_.dmat_d
        self.N = self.pde_solve_.N
        self.Integrate=self.pde_solve_.Integrate
        
        # Run parameter checks
        self._check_params()
        # initialize peq
        self.peq_ = self.peq_model_(self.x_d_, self.w_d_)
        # initialize D
        self.D_ = self.D0
        #initialize Firing Rates
        num_neuron=len(self.firing_model_)
        if self.num_neuron!=num_neuron:
            print (f'Warining: {num_neuron} neurons was found, while the parameter num_neuron was set to {self.num_neuron}! Setting num_neuron to the correct value.')
            self.num_neuron=num_neuron
        self.fr_=np.empty((self.N,self.num_neuron),dtype='float64')
        for i in range(self.num_neuron):
            self.fr_[:,i]=self.firing_model_[i](self.x_d_)
        self.converged_ = False
    
    
    
    
    
    
    def _check_params(self):
        """Check parameters of EM class initialization
        """
       
        # Check num_neuron
        if not isinstance(self.num_neuron,int) or self.num_neuron <1:
            raise ValueError("Number of neurons is not a positive integer")
        
        #If firing_model consists of 1 neuron and passed as dictionary, convert into list
        if self.num_neuron==1 and isinstance(self.firing_model,dict):
            self.firing_model=[self.firing_model]
        
        #firing model should be a list
        if not isinstance(self.firing_model,list):
            raise ValueError("Input firing rate model as list")
        self.firing_model_=[]
        
        #Check firing_rate_model and evaluate firing rates
        for i,fr in enumerate(self.firing_model):
            if callable(fr):
                self.firing_model_.append(fr)
            elif 'model' in fr and fr['model'] in self._firing_model_types:
                #If there are any parameters
                if len(fr['params'].values())>0:
                    #number of neurons is equal to number of parameters
                    n_neurons=np.asarray(next(iter(fr['params'].values()))).size 
                    #Convert params into list of dictionaries, one for each of the neurons
                    if n_neurons>1:
                        params_list=[dict((keys,fr['params'][keys][i]) for keys in fr['params'].keys()) for i in range(n_neurons)]
                    else:
                        params_list=[dict((keys,fr['params'][keys]) for keys in fr['params'].keys())]
                    
                    #add lambda function for each neuron
                    for iFun in range(n_neurons):
                        self.firing_model_.append(lambda x, i=i, iFun=iFun, params_list=params_list: self._firing_model_types[fr['model']](x, **params_list[iFun])) 
                else:
                    self.firing_model_.append(lambda x, i=i: self._firing_model_types[fr['model']](x))
            else:
                raise ValueError("Firing rate model for neuron {}: unknown or not provided".format(i))         
        
        # Check peq model
        if callable(self.peq_model):
            self.peq_model_ = self.peq_model
        else:
            if self.peq_model['model'] in self._peq_model_types:
                self.peq_model_ = lambda x, w: self._peq_model_types[self.peq_model['model']](x, w, **self.peq_model['params'])
            else:
                raise ValueError("peq_model type should be one of %s or callable, "
                                 "%s was given."
                                 % (self._peq_model_types.keys(), self.peq_model['model']))

        # Check D0 value
        if self.D0 <= 0.:
            raise ValueError("D0 must be positive.")

        # Force verbose type to bool
        self.verbose = bool(self.verbose)

    def _check_data(self, data):
        """Check data
        """
        
        if (not isinstance(data, np.ndarray)):
            raise ValueError("Data must be an array of spike-time arrays and indecies"
                             "of neurons which fired for each trial.")
        if (not len(data.shape)==2):
            raise ValueError("Data must be an array with the shape (num_trial,2). "
                             "The shape of the provided array was {} instead.".format(data.shape))
        elif (not data.shape[1]==2):
            raise ValueError("Data must be an array with the shape (num_trial,2). "
                             "The shape of the provided array was {} instead.".format(data.shape))
        # number of sequencies in the data
        num_seq = data.shape[0]
        if (num_seq==0):
            raise ValueError("Empty list of spike-time arrays is provided, need some data to work with.")
        # now check each spike-time array
        for iSeq in range(num_seq):
            # check sequence for each trial
            self._check_sequence(data[iSeq,:])


    def _check_sequence(self, sequence):
        """Check ISIs sequence
        """
        seq = sequence[0]
        if (not isinstance(seq, np.ndarray)):
            raise ValueError("Interspike interval data must be provided as a numpy array.")
        elif (not seq.dtype == np.double):
            raise ValueError("Interspike intervals must be double-precision numbers.")
        elif np.any(seq < 0.0):
            raise ValueError("Interspike interval data must be non-negative.")
        # check the data for index of neurons which fired
        ind_seq = sequence[1]
        if (not isinstance(ind_seq, np.ndarray)):
            raise ValueError("Indecies of neurons which fired must be provided as a numpy array.")
        elif (not ind_seq.dtype == np.int):
            raise ValueError("Indecies of neurons which fired must be integers.")
        elif (np.any(ind_seq < 0.0) or np.any(ind_seq >= self.num_neuron)):
            raise ValueError("Indecies of neurons which fired must be within the range of number of neurons.")
        # check that inter-spike interval and indeces arrays are the same length
        if (not seq.shape==ind_seq.shape):
            raise ValueError("Arrays of interspike intervals and of indecies of neurons which fired "
                             "must have the same length.")
    
    def _check_optimization_options(self, optimizer, options, save_iter):
        """Check optimization options
        """
        if optimizer=='GD':
            if 'max_iteration' not in options:
                options['max_iteration']=100
            elif not np.issubdtype(type(options['max_iteration']),np.integer) or options['max_iteration']<1:
                raise ValueError('Invalid max_iteration option')
            
            if 'loglik_tol' not in options:
                options['loglik_tol']=0
            else:
                if not isinstance(options['loglik_tol'], numbers.Number):
                    raise ValueError('loglik_tol must be a positive number')
            
            if 'sim_start' not in options: 
                options['sim_start']=0
            elif not np.issubdtype(type(options['sim_start']),np.integer) or options['sim_start']<0:
                    raise ValueError('sim_start has to be a non-negative integer')
            
            if 'sim_id' not in options:
                options['sim_id']=''
            
            
            if 'gamma' in options:
                params_to_opt=list(options['gamma'].keys())
                if not all (i in ['F', 'D'] for i in iter(params_to_opt)):
                    raise ValueError ('Unknown optimization parameters encountered')
                if not all ((isinstance(options['gamma'][i], numbers.Number) or isinstance(options['gamma'][i], list)) for i in iter(options['gamma'].keys())):
                    raise ValueError ('Entries in gamma must correspond to params_to_opt enteries')
            else:
                raise ValueError('gamma must be specified in options for {} optimizier'.format(optimizer))
                
            if 'schedule' in options:
                #Sort schedule 
                options['schedule'].sort()
                
                #Remove duplicates
                options['schedule']=options['schedule'][np.insert(np.diff(options['schedule']).astype(np.bool),0,True)]
                
                #Needed for schedule adjustment
                new_sim=True if options['sim_start']==0 else False
                max_iteration = options['max_iteration']
                
                #Save initial parameters (zero iteration) by default if new_sim=True
                if options['schedule'][0]!=0 and new_sim:
                    options['schedule']=np.insert(options['schedule'],0,0)
                elif options['schedule'][0]==0 and (not new_sim):
                    options['schedule']=options['schedule'][1:] 
        
                #Check if last iteration to be saved coincides with max_iteration
                if options['schedule'][-1]<max_iteration+options['sim_start']+new_sim-1:
                    print("Setting max_iteration to {}".format(options['schedule'][-1]-options['sim_start']+1-new_sim))
                    options['max_iteration']=options['schedule'][-1]-options['sim_start']+1-new_sim
                    options['max_iteration']=int(options['max_iteration'])
                elif options['schedule'][-1]>max_iteration+options['sim_start']:
                    raise ValueError("Iteration #{} exceeds max_iterations".format(options['schedule'][-1]))
            else:
                new_sim=True if options['sim_start']==0 else False
                options['schedule']=np.arange(options['sim_start'], options['max_iteration']+options['sim_start']+new_sim)
            
            if save_iter is not None:
                if 'path' not in save_iter:
                    print('Warning: path is not specified! No intermediate savings will be made')
                    save_iter=None
                elif not isinstance(save_iter['path'],str):
                    print('Warning: path is not specified! No intermediate savings will be made')
                    save_iter=None
                elif 'stride' not in save_iter:
                    save_iter['stride']=options['max_iteration']
                elif save_iter['stride']>options['max_iteration']:
                    print('Stride exceeds maximum value! No intermediate savings will be made')
                    save_iter=None
            else:
                save_iter=None
            
            