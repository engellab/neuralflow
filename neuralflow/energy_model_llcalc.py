# -*- coding: utf-8 -*-
"""This is a part of neuralflow package/EnergyModel class.
This source file contains functions related to log-likelihood and it's derivatives calculation (and the supporting functions)"""

import numpy as np
from scipy import linalg
from . import c_get_gamma
from collections.abc import Iterable
from copy import deepcopy


def _get_loglik_data(self, data, metadata=None, peq=None, rho0=None, D=None, fr=None, mode='loglik', grad_list=None, EV_solution=None):
    """Calculates loglikelihood and/or gradients
    See Supplementary Information for M. Genkin, T. A. Engel, Nat Mach Intell 2, 674–683 (2020), and M. Genkin, O. Hughes, T. A. Engel, Nat Commun 12, 5986 (2021).


    Parameters
    ----------
    data : numpy array (N,2), dtype=np.ndarray
        Spike data packed as numpy array of the size (N,2), where each elements is a 1D array.
        See the docstring to _GD_optimization for details.
    metadata : dictionary
        Metadata dictionary, see the docstring to _GD_optimization for details. The default is None.
    peq : numpy array, dtype=float
        Equilirium probability distribution that defines model potential.
    rho0 : numpy array, dtype=float
        Scaled initial probaiblity distribution, normalized by sqrt(peq): rho0=p0/np.sqrt(peq).
    D : float
        Diffusion (noise) magnitude.
    fr : numpy array, dtype=float
        firing rate funcitons for all the neurons. 2D array, where number of columns is equal to the number of neurons.
    mode : ENUM('loglik','gradient','gradient_eq')
        Determines whether only the likelihood should be calculated (only forward pass), or the likelihood and
        variational derivatives (forward and backward passes)
              'loglik' - calculate likelihood (grad_list is ignored)
              'gradient' - calculate gradients (list of gradients should be supplied in grad_list)
              'gradient_eq' - same as 'gradient', but assuming each trial starts from the equilibrium distribution of the latent states.
    grad_list : list
        List of parameters for which gradients should be calculated (only for the 'gradient' mode). Supported options are 'F', 'F0', and 'D' .
        The default is None.
    EV_solution : dicionary
        Dictionary with the solution of the eigenvector-eigenvalue (EV) problem. The format is {'lQ':lQ, 'QxOrig':QxOrig, 'Qx':Qx, 'lQd':lQd, 'Qd': Qd}
        If not provided, will be calculated. The default is None.

    Returns
    -------
    results : dictionary
        Dictionary with the results. Possible entries are 'loglik', 'D', 'F', 'F0'
    """

    # number of trials
    num_seq = data.shape[0]

    # Default peq,D,fr are self
    peq = self.peq_ if peq is None else peq
    D = self.D_ if D is None else D
    fr = self.fr_ if fr is None else fr

    # set rho0 to sqrt(peq) if not provided - initial distribution is assumed to be equilibrium
    if rho0 is None:
        rho0 = np.sqrt(peq)
        if mode == 'gradient':
            mode = 'gradient_eq'

    # Initialize results_all, which will accumulate the gradients and logliklihoods over trials, or record posteriors of latent states on each trial
    results_all = {}
    results_all['loglik'] = 0
    if grad_list is not None:
        for grad in grad_list:
            results_all[grad] = 0

    # Solve EV problaem if the solution is not provided,
    if EV_solution is None:
        EV_solution = self._get_EV_solution(peq, D, fr)

    # For convinience convert metadata into list if needed
    if metadata is None:
        metadata = {"last_event_is_spike": np.full(num_seq, None), "absorption_event": np.full(num_seq, None)}
    elif type(metadata) is dict:
        metadata = {"last_event_is_spike": metadata["last_event_is_spike"] if isinstance(metadata["last_event_is_spike"], Iterable) else np.full(num_seq, metadata["last_event_is_spike"]),
                    "absorption_event": metadata["absorption_event"] if isinstance(metadata["absorption_event"], Iterable) else np.full(num_seq, metadata["absorption_event"])}

    for iSeq in range(num_seq):
        # Get ISI and the corresponding neuron ids
        data_trial = deepcopy(data[iSeq, :])
        last_event_is_spike = metadata["last_event_is_spike"][iSeq] if metadata["last_event_is_spike"][iSeq] is not None else False
        absorption_event = metadata["absorption_event"][iSeq] if metadata["absorption_event"][iSeq] is not None else True if self.boundary_mode == 'absorbing' else False

        # Delete the last propagation in latent space if we are not using it
        if last_event_is_spike and data_trial[1][-1] == -1:
            data_trial[0] = data_trial[0][:-1]
            data_trial[1] = data_trial[1][:-1]

        results = self._get_loglik_seq(data_trial, peq, rho0, D, fr, mode, grad_list, EV_solution, last_event_is_spike, absorption_event)
        # Sum the results across trials
        if grad_list is not None:
            for grad in grad_list:
                results_all[grad] += results[grad]
        results_all['loglik'] += results['loglik']

    return results_all


def _get_EV_solution(self, peq, D, fr):
    """Solve Eigenvector-eigenvalue problem. Needed for likelihood/gradients calculation


    Parameters
    ----------
    peq : numpy array, dtype=float
        The equilibrium probability density evaluated on SEM grid. The default is self.peq_
    D : float
        Noise intensity. The default is self.D_
    fr : numpy array, dtype=float
        2D array that contains firing rate functions for each neuron evaluated on SEM grid. The default is self.fr_

    Returns
    -------
    dict
        Dictionary with the following entries:
        lQ : numpy array, dtype=float
            Eigenvalues of H0, 1D array of floats
        QxOrig : numpy array, dtype=float
            Scaled eigenvectors of H0 (divided by sqrt(peq)), 2D array where each column is an eigenvector
        Qx : numpy array, dtype=float
            Eigenvectors of H0, 2D array where each column is an eigenvector
        lQd : numpy array, dtype=float
            Eigenvalues of H, 1D array of floats
        Qd : numpy array, dtype=float
            Eigenvectors of H in the basis of H0, 2D array where each column is an eigenvector
    """
    fr_cum = np.sum(fr, axis=1)
    lQ, QxOrig, Qx, lQd, Qd = self.pde_solve_.solve_EV(peq=peq, D=D, w=peq, mode='hdark', fr=fr_cum, Nv=self.Nv)
    return {'lQ': lQ, 'QxOrig': QxOrig, 'Qx': Qx, 'lQd': lQd, 'Qd': Qd}


def _get_loglik_seq(self, data, peq, rho0, D, fr, mode, grad_list, EV_solution, last_event_is_spike, absorption_event):
    """This function calculates loglik/gradients for a single trial.

    Parameters
    ----------
    data : numpy array (N,2), dtype=np.ndarray.
        Spike data packed as numpy array of the size (N,2), where each elements is a 1D array.
        See the docstring to _GD_optimization for details.
    peq : numpy array, dtype=float
        Equilirium probability distribution that defines model potential
    rho0 : numpy array, dtype=float
        Scaled initial probaiblity distribution, normalized by sqrt(peq): rho0=p0/np.sqrt(peq).
    D : float
        Diffusion (noise) magnitude
    fr : numpy array, dtype=float
        firing rate funcitons for all the neurons. 2D array, where number of columns is equal to the number of neurons
    mode : ENUM('loglik','gradient','gradient_eq')
        Determines whether only the likelihood should be calculated (only forward pass), or the likelihood and
        variational derivatives (forward and backward passes)
              'loglik' - calculate likelihood (grad_list is ignored).
              'gradient' - calculate gradients (list of gradients should be supplied in grad_list).
              'gradient_eq' - same as 'gradient', but assuming each trial starts from the equilibrium distribution of the latent states.
    grad_list : list
        List of parameters for which gradients should be calculated (only for gradient mode). Supported options are 'F', 'F0' and 'D'.
    EV_solution : dictionary
        Solution of EV problem:
            lQ : numpy array, dtype=float
                Eigenvalues of H0, 1D array of floats
            QxOrig : numpy array, dtype=float
                Scaled eigenvectors of H0, 2D array where each column is an eigenvector
            Qx : numpy array, dtype=float
                Eigenvectors of H0, 2D array where each column is an eigenvector
            lQd : numpy array, dtype=float
                Eigenvalues of H, 1D array of floats
            Qd : numpy array, dtype=float
                Eigenvectors of H in the basis of H0, 2D array where each column is an eigenvector
    last_event_is_spike : bool
        If true, trial termination time will be ignored (even if recorded). Otherwise, trial termination time will be used.
    absorption_event : bool
        If true, absorption operator will be applied in the end of the loglikelihood chain.


    Returns
    -------
    dictionary
        Dictionary with the results. Possible entries are 'loglik', 'F','F0','D'

    """
    # Extract ISI and neuron_id for convenience
    seq, nid = data

    # Sequence length
    S_Total = len(seq)

    # Transformation from SEM to H basis
    Qxd = EV_solution["Qx"].dot(EV_solution["Qd"])
    if absorption_event:
        # Transformation from H0 to H basis
        Qxd_hzf = EV_solution["Qd"]

    # Get the spike operator in the H basis.
    Sp = np.zeros((self.Nv, self.Nv, self.num_neuron), dtype=peq.dtype)
    for i in range(self.num_neuron):
        Sp[:, :, i] = (Qxd.T * fr[:, i] * self.w_d_).dot(Qxd)

    # Initialize the atemp and btem for the forward and backwards passes.
    atemp = Qxd.T.dot(self.w_d_ * rho0)
    btemp = Qxd.T.dot(self.w_d_ * np.sqrt(peq))

    # Normalization coefficients
    anorm = np.zeros(S_Total + 3, dtype=peq.dtype) if absorption_event else np.zeros(S_Total + 2, dtype=peq.dtype)
    anorm[0] = linalg.norm(atemp)
    atemp /= anorm[0]

    # Store alphas for gradient calculation
    alpha = np.zeros((self.Nv, anorm.size), dtype=peq.dtype)
    alpha[:, 0] = atemp

    # Precalculate exp(-lambda_i Delta t) propagation matrix
    dark_exp = np.exp(np.outer(-EV_solution["lQd"], seq))

    # Absorption operator
    if absorption_event:
        absorption_operator = Qxd_hzf.T.dot(np.diag(EV_solution["lQ"])).dot(Qxd_hzf)

    # Calcuate the alpha vectors (forward pass)
    for i in range(1, S_Total + 1):
        # Propagate forward in latent space
        atemp *= dark_exp[:, i - 1]

        # Spike observation
        if i != S_Total or last_event_is_spike:
            atemp = atemp.dot(Sp[:, :, nid[i - 1]])

        # Calculate l2 norm (np.linalg.norm(atemp) is faster than np.sqrt(np.sum(atemp**2)))
        anorm[i] = np.linalg.norm(atemp)

        # Normalize alpha
        atemp /= anorm[i]

        # save the current alpha vector
        alpha[:, i] = atemp

    # Apply absorption operator
    if absorption_event:
        atemp = atemp.dot(absorption_operator)
        anorm[S_Total + 1] = np.sqrt(np.sum(atemp**2))
        atemp /= anorm[S_Total + 1]
        alpha[:, S_Total + 1] = atemp

    # The last anorm coefficient is the product of alpha_N and beta_N
    anorm[-1] = btemp.dot(atemp)
    btemp /= anorm[-1]

    # compute negative log-likelihood
    ll = -np.sum(np.log(anorm))

    if mode == 'likelihood':
        return {'loglik': ll}

    G = np.zeros((self.Nv, self.Nv), dtype=peq.dtype)

    # Backward pass also starts with absorption
    if absorption_event:
        btemp /= anorm[S_Total + 1]
        # Contribution to G function
        Gabs = np.outer(alpha[:, i], btemp)
        # Propagate and normalise
        btemp = (absorption_operator).dot(btemp)

    # Backwards pass
    for i in reversed(range(S_Total)):
        dt = seq[i]
        tempExp = dark_exp[:, i]

        # Scaling
        btemp /= anorm[i + 1]
        # Spike emission
        if i != S_Total - 1 or last_event_is_spike:
            btemp = Sp[:, :, nid[i]].dot(btemp)
        # G function
        c_get_gamma.getGamma0(
            G, self.Nv, EV_solution["lQd"], tempExp, alpha, btemp, dt, i)
        # Propagation in latent space
        btemp *= tempExp

    # add contribution to G from absorption_event
    if absorption_event:
        G -= Gabs

    # Transform G to HO basis for convinience
    G0 = EV_solution["Qd"].dot(G).dot(EV_solution["Qd"].T)

    QxOrig = EV_solution["QxOrig"]
    Qxdx = self.dmat_d_.dot(QxOrig)

    results = {}
    results['loglik'] = ll

    if 'F' in grad_list:
        dPhi = np.sum(QxOrig * Qxdx.dot(G0), 1) + np.sum(Qxdx * QxOrig.dot(G0), 1)
        # boundary term depends on whether or not we start a trajectory from peq or p0
        if mode == 'gradient_eq':
            ABF = self.Integrate(peq - 0.5 * Qxd.dot(atemp / anorm[-1] + btemp / anorm[0]) * np.sqrt(peq))
        else:
            ABF = 0.5 * self.Integrate(Qxd.dot(btemp / anorm[0]) * rho0 - Qxd.dot(atemp / anorm[-1]) * np.sqrt(peq))
        results['F'] = -0.5 * D * peq * dPhi - ABF
    if 'D' in grad_list:
        d2Phi = np.sum(Qxdx * Qxdx.dot(G0), 1)
        results['D'] = np.sum(self.w_d_ * peq * d2Phi)
    if 'F0' in grad_list:
        results['F0'] = - \
            self.Integrate(rho0 * np.sqrt(peq) - rho0 * Qxd.dot(btemp / anorm[0]))
    return results
