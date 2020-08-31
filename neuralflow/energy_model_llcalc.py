# -*- coding: utf-8 -*-

# This is a part of EnergyModel class
# This source file contains functions related to log-likelihood and it's derivatives calculation (and the supporting functions)

import numpy as np
from scipy import linalg
from . import c_get_gamma


def _get_loglik_data(self, data, peq, D, fr, mode, grad_list=None, EV_solution=None):
    """Calculates loglikelihood and/or gradients 
    See Supplementary Information 1.2, 1.3 for details.
    
    
    Parameters
    ----------
    data : numpy array (N,2),dtype=np.ndarray. 
        Spike data packed as numpy array of the size (N,2), where each elements is a 1D array of floats. 
        N is the number of trials, and for each trial the first column contains inter spike intervals (ISIs), 
        and the second column contains the corresponding neuronal indices (in the paper only one neural 
        response was always used, so the neuronal indices are all set to 0). ISIs are represented as 1D arrays
        of floats, and neuronal indices as 1D array of integers.
    peq : numpy array, dtype=float
        Equilirium probability distribution that defines model potential, see Supplementary Section 1.1   
    D : float
        Diffusion(noise) magnitude
    fr : numpy array, dtype=float
        firing rate funcitons for all the neurons. 2D array, where number of columns is equal to the number of neurons
    mode : ENUM('loglik','gradient') 
        Determines whether only the likelihood should be calculated (only forward pass), or the likelihood and
        variational derivatives (forward and backward passes)
              'loglik' - calculate likelihood (grad_list is ignored)
              'gradient' - calculate gradients (list of gradients should be supplied in grad_list)
    grad_list : list
        List of parameters for which gradients should be calculated (only for gradient mode). Supported options are 'F' and 'D' . The default is None.  
    EV_solution : dicionary
        Dictionary with the solution of eigenvector-eigenvalue (EV) problem. The format is {'lQ':lQ, 'QxOrig':QxOrig, 'Qx':Qx, 'lQd':lQd, 'Qd': Qd}
        If not provided, will be calculated. The default is None. 
       
    Returns
    -------
    results : dictionary
             Dictionary with the results. Possible entries are 'loglik', 'D', 'peq'
            
    """
    # number of sequencies in the data
    num_seq = data.shape[0]
    
    #Initialize results_all, which will accumulate gradients and logliklihoods over trials
    results_all={}
    results_all['loglik']=0
    if grad_list is not None:
        for grad in grad_list:
            results_all[grad]=0
    
    #Solve EV problem
    if EV_solution is None:
        EV_solution = self._get_EV_solution(peq,D,fr)
    
    #Calculate logliks and gradients on each trial and sum then up
    for iSeq in range(num_seq):
        #Get ISI and corresponding neuron ids
        data_trial = data[iSeq,:] 
        results = self._get_loglik_seq(data_trial, peq, D, fr, mode, grad_list, **EV_solution)
        if grad_list is not None:
            for grad in grad_list:
                results_all[grad]+=results[grad]
        results_all['loglik']+=results['loglik']
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
            Scaled eigenvectors of H0, 2D array where each column is an eigenvector
        Qx : numpy array, dtype=float
            Eigenvectors of H0, 2D array where each column is an eigenvector
        lQd : numpy array, dtype=float
            Eigenvalues of H, 1D array of floats
        Qd : numpy array, dtype=float
            Eigenvectors of H in the basis of H0, 2D array where each column is an eigenvector
    """
    fr_cum=np.sum(fr,axis=1)
    lQ, QxOrig, Qx, lQd, Qd = self.pde_solve_.solve_EV(peq=peq, D=D, w=peq, mode='hdark', fr=fr_cum, Nv=self.Nv)
    return {'lQ':lQ, 'QxOrig':QxOrig, 'Qx':Qx, 'lQd':lQd, 'Qd': Qd}
    
def _get_loglik_seq(self, data, peq, D, fr, mode, grad_list, lQ, QxOrig, Qx, lQd, Qd):
    """This function calculates loglik/gradients for a single trial.
    
    Parameters
    ----------
    data : numpy array (N,2),dtype=np.ndarray. 
        Spike data packed as numpy array of the size (N,2), where each elements is a 1D array of floats. 
        N is the number of trials, and for each trial the first column contains inter spike intervals (ISIs), 
        and the second column contains the corresponding neuronal indices (in the paper only one neural 
        response was always used, so the neuronal indices are all set to 0). ISIs are represented as 1D arrays
        of floats, and neuronal indices as 1D array of integers.
    peq : numpy array, dtype=float
        Equilirium probability distribution that defines model potential, see Supplementary Section 1.1   
    D : float
        Diffusion(noise) magnitude
    fr : numpy array, dtype=float
        firing rate funcitons for all the neurons. 2D array, where number of columns is equal to the number of neurons
    mode : ENUM('loglik','gradient') 
        Determines whether only the likelihood should be calculated (only forward pass), or the likelihood and
        variational derivatives (forward and backward passes)
              'loglik' - calculate likelihood (grad_list is ignored)
              'gradient' - calculate gradients (list of gradients should be supplied in grad_list)
    grad_list : list
        List of parameters for which gradients should be calculated (only for gradient mode). Supported options are 'F' and 'D' .  
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
    
    Returns
    -------
    dictionary
        Dictionary with the results. Possible entries are 'loglik', 'D', 'peq'

    """
    #Extract ISI and neuron_id for convenience
    seq=data[0]
    nid=data[1]
    
    #Number of spikes
    S_Total = len(seq)
    
    # transformation matrix from SEM basis to the Hdark basis
    Qxd = Qx.dot(Qd)
    
    # Get the spike operator in the Hdark basis for each neural responce
    Sp = np.zeros((self.Nv,self.Nv,self.num_neuron))
    for i in range (self.num_neuron):
        Sp[:,:,i]= (Qxd.T*fr[:,i] * self.w_d_).dot(Qxd)
    
    # Initialize all of the atemp values.
    atemp = np.zeros(self.Nv)
    # equilibrium distrubution is the principle eigenvector of H0
    atemp[0] = 1 
    atemp = Qd.T.dot(atemp) # transform to Hdark
    
    # Terminal vector is the same as initial, see Supplementary Info
    btemp = np.copy(atemp)       
    
    # Scaling coefficints. This is to prevent numerical underflow. Similar trick is used with HMMs
    anorm = np.zeros(S_Total + 2)
    anorm[0] = linalg.norm(atemp)
    atemp /= anorm[0]
    
    alpha = np.zeros((self.Nv, S_Total + 1))
    alpha[:, 0] = atemp
    
    #Precalculate dark propogation matrix
    dark_prop=np.exp(np.outer(-lQd, seq))
    
    # Calcuate the alpha vectors (forward pass)
    for i in range(1, S_Total + 1):
        # Propagete forward in latent space
        atemp *= dark_prop[:,i-1]
        
        # Spike operator
        atemp = Sp[:,:,nid[i-1]].dot(atemp)
        
        # Calculate l2 norm (scaling)
        anorm[i]= np.sqrt(np.sum(atemp**2))
        
        # Normalize alpha
        atemp /= anorm[i]
        
        # save the current alpha vector
        alpha[:, i] = atemp
        
    anorm[-1] = btemp.dot(atemp)
    btemp /= anorm[-1]
    
    # compute negative log-likelihood
    ll = -np.sum(np.log(anorm))
    
    if mode == 'likelihood':
        return {'loglik':ll}
    
    tempExp = np.zeros(self.Nv)
    G = np.zeros((self.Nv, self.Nv))
    
    # Calculate beta vectos (backward pass)
    for i in reversed(range(S_Total)):
        dt = seq[i]
        tempExp=dark_prop[:,i]
        
        #Propagate beta back (scaling - spike emission - propagation in latent space) and calculate G function
        btemp /= anorm[i+1]
        btemp = Sp[:,:,nid[i]].dot(btemp)
        c_get_gamma.getGamma0(G, self.Nv, lQd, tempExp, alpha, btemp, dt, i)
        btemp *= tempExp
    
    # Transform G to HO basis for convinience
    G0 = Qd.dot(G).dot(Qd.T)
    
    # first calculate derivatives of the eigenvectors
    Qxdx = self.dmat_d_.dot(QxOrig)
    Qxdx[:, 0] = 0 #With Neumann or Dirichlet boundary conditions d(Phi_i(x)Phi)j(x))/dx is always zero on the boundary
    
    results={}
    results['loglik']=ll
       
    if 'F' in grad_list:
        dPhi = np.sum(QxOrig * Qxdx.dot(G0), 1) + np.sum(Qxdx * QxOrig.dot(G0), 1) 
        ABF=self.Integrate(peq-0.5*Qxd.dot(atemp/anorm[-1]+btemp/anorm[0])*np.sqrt(peq))
        results['F'] = -0.5 * D * peq * dPhi - ABF #See Eq.(35) in Supplementary Notes
    if 'D' in grad_list:
        d2Phi = np.sum(Qxdx * Qxdx.dot(G0), 1)
        results['D'] = np.sum(self.w_d_ * peq * d2Phi )  
        
    return results    