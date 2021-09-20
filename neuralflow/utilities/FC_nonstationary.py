#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
FC_nonstationary module
--------------------------------------------------

Below are the complimentary functions for Model selection based on feature complexity analysis for non-stationary data.
See Genkin, Hughes, Engel, arxiv 2020 paper for details.
This module is optional and should be imported separately: from neuralflow.utilities import FC_nonstationary.
"""
import numpy as np

def FeatureConsistencyAnalysis(data1, data2, em, iteration_indices, FC_stride):
    """ Calculate JS divergence and feature complexities for two sequences of models data1 and data2.
    
    This function calculate JS(FC) curve for the case when only the model potential is optimized (and the rest of the parameters are fixed).
    
    Parameters
    ----------
    data1 : dictionary
        Dictionary with peqs fitted on datasample 1 (the 1st sequence of fitted models).
    data2 : dictionary
        Dictionary with peqs fitted on datasample 2 (the second sequence of fitted models).
    em : EnergyModel
        An instance of energymodel class used for fitting (that contains common D, p0, and fr functions).
    iteration_indices : numpy array
        Indices of peqs in data1/data2 arrays which will be used for the analsys (to accelerate the computation, 
        one may only consider the models at a subset of iterations, rather then all availible models).
    FC_stride : int
        This parameter determines how many models from the sequence 2 will be compared with the model from the sequence 1 at each level of 
        feature complexity. The number of comparison at each level of feature complexity is 2*FC_stride+1.

    Returns
    -------
    FC : numpy array
        Feature complexities.
    JS : numpy array
        JS divergences between the two models at each level of feature complexity.
    inds1 : numpy array
        Indices of the models at each level of feature complexity in data1 array.
    inds2 : numpy array
        Indices of the models at each level of feature complexity in data1 array.

    """
    # Calculate feature complexities, eigenvalues and eigenvectors for the selected models
    FCs_array1, lQ_array1, Qx_array1=FeatureComplexities(data1,em,iteration_indices)
    FCs_array2, lQ_array2, Qx_array2=FeatureComplexities(data2,em,iteration_indices)

    # Preallocate JS divergence array 
    JS = np.zeros(FCs_array1.size)

    # For the second sequence, mid_inds2 will contain index of the model for each feature complexity in data2.
    inds2 = np.zeros_like(JS)

    for i in range(FCs_array1.size):
        # We only consider 1 model from the sequence of models optimized on data sample 1.
        FC1_ind = i
        
        # Find the index of a model in the second sequence FCs_array2 that have the closest feature complexity  
        # to the model from the first sequence 
        ind_seq_2 = np.argmin(np.abs(FCs_array1[FC1_ind]-FCs_array2))
        
        # Select the indices of 2*FC_stride-1 models from the second sequence of models around the index ind_seq_2
        FC2_ind = np.array(np.arange(max(0,ind_seq_2-FC_stride),min(FCs_array2.size-1,ind_seq_2+1+FC_stride)))    
        
        #Compute JSD for each pair of models selected from the sequences 1 and 2.
        JS_cur = np.zeros(FC2_ind.size)
        
        for i2, ind2 in enumerate(FC2_ind):
            peq_ind1=iteration_indices[FC1_ind]
            peq_ind2=iteration_indices[ind2]
            JS_cur[i2]=JS_divergence_tdp(data1['peqs'][...,peq_ind1],em.D_,em.p0_,
                                            data2['peqs'][...,peq_ind2],em.D_,em.p0_,
                                            em.w_d_, lQ_array1[FC1_ind,:], Qx_array1[FC1_ind,:], 
                                            lQ_array2[ind2,:], Qx_array2[ind2,:],1,10)
        
        # Find the index of model2 in data2 
        m2 = np.argmin(JS_cur)
        inds2[i] = iteration_indices[FC2_ind[m2]]
        JS[i] = JS_cur[m2]
    # For the first sequence the indices of the models are equal to the selected iterations
    inds1 = iteration_indices[np.arange(FCs_array1.size)]
    return FCs_array1,  JS, inds1, inds2
        

def FeatureComplexity(em, peq=None, p0=None, D=None):
    """Calculate feature complexity for a single model specified by (peq,D,p0).


    Parameters
    ----------
    em : EnergyModel
        An instance of the EnergyModel class.
    peq : numpy array
        Equilibirum probability distribution.
    p0 : numpy array
        Distribution of the initial latent states.
    D : float
        Noise magnitude

    Returns
    -------
    FC : float
        Feature complexity.

    """

    if peq is None:
        peq = em.peq_
    if p0 is None:
        p0 = em.p0_
    if D is None:
        D = em.D_

    Seq = np.sum(p0 * np.log(p0) * em.w_d_)
    Force = em.dmat_d_.dot(np.log(peq))
    Fs = D / 4 * Force**2
    lQ, _, Qx = em.pde_solve_.solve_EV(
        peq, D, q=None, w=peq, mode='h0', fr=None, Nv=em.Nv)
    Fd = Qx.T.dot(np.diag(em.w_d_)).dot(Fs * np.sqrt(peq))
    rho0d = Qx.T.dot(np.diag(em.w_d_)).dot(em.p0_ / np.sqrt(peq))
    S2 = np.sum(Fd * rho0d / lQ)
    return Seq + S2


def FeatureComplexities(results, em, iterations):
    """Calculate feature complexities of the model at specified iterations, and also eigenvalues and eigenvectors of the H0 operator.


    Parameters
    ----------
    results : dictionary
        Dictionary with the results (fitted models)
    em : EnergyModel
        An instance of the EnergyModel class.
    iterations : numpy array
        Iteration numbers on which feature complexities/EVs/EVVs should be calculated.

    Returns
    -------
    FCs_array : numpy array
        A 1D array of feature complexities.
    lQ_array : numpy array
        A 2D array of H0 eigenvalues.
    Qx_array : numpy array
        A 3D array of H0 eigenvectors.

    """
    lQ_array = np.zeros((iterations.size, em.Nv))
    Qx_array = np.zeros((iterations.size, em.N, em.Nv))
    FCs_array = np.zeros((iterations.size))
    p0 = em.p0_
    D = em.D_
    Seq = np.sum(p0 * np.log(p0) * em.w_d_)
    for j, i in enumerate(iterations):
        peq = results['peqs'][..., i]
        Force = em.dmat_d_.dot(np.log(peq))
        Fs = D / 4 * Force**2

        lQ, _, Qx = em.pde_solve_.solve_EV(
            peq, D, q=None, w=peq, mode='h0', fr=None, Nv=em.Nv)
        lQ_array[j, ...] = lQ.copy()
        Qx_array[j, ...] = Qx.copy()

        Fd = Qx.T.dot(np.diag(em.w_d_)).dot(Fs * np.sqrt(peq))
        rho0d = Qx.T.dot(np.diag(em.w_d_)).dot(em.p0_ / np.sqrt(peq))
        S2 = np.sum(Fd * rho0d / lQ)
        FCs_array[j] = Seq + S2

    return FCs_array, lQ_array, Qx_array


def JS_divergence(p1, p2, weights, mode='normalized'):
    """Calculate JS divergence between two distributions


    Parameters
    ----------
    p1 : numpy array
        The first distribution, 1D array
    p2 : numpy array
        The second distribution, 1D array
    weights: numpy array
        1D array of SEM weights
    mode : ENUM('normalized','unnormalized')
        If normalized, both distribution are assumed to be normalized (integrate to 1). Otherwise, boundary term will be accounted for.

    Returns
    -------
    JS : float
        JS divergence

    """
    p1 = np.maximum(p1, 10**-10)
    p2 = np.maximum(p2, 10**-10)
    M = 0.5 * (p1 + p2)
    if mode == 'normalized':
        return 0.5 * (np.sum(weights * p1 * np.log(p1 / M)) + np.sum(weights * p2 * np.log(p2 / M)))
    else:
        I1 = 1 - np.sum(weights * p1)
        I2 = 1 - np.sum(weights * p2)
        return 0.5 * (np.sum(weights * p1 * np.log(p1 / M)) + np.sum(weights * p2 * np.log(p2 / M)) + I1 * np.log(2 * I1 / (I1 + I2)) + I2 * np.log(2 * I2 / (I1 + I2)))


def JS_divergence_tdp(peq1, D1, p01, peq2, D2, p02, weights, lQ1, Qx1, lQ2, Qx2, terminal_time=0.5, number_of_samples=5):
    """Calculate JS divergence between two time-dependent probability distributions that comes from two different non-stationary Langevin dynamics.


    Parameters
    ----------
    peq1 : numpy array
        1D array of peq values (that defines Langevin potential) for the first dynamics.
    D1 : float
        Noise magnitude for the first dynamics.
    p01 : numpy array
        1D array of p0 distribution for the first dynamics.
    peq2 : numpy array
        1D array of peq values (that defines Langevin potential) for the second dynamics.
    D2 : float
        Noise magnitude for the second dynamics.
    p02 : numpy array
        1D array of p0 distribution for the second dynamics.
    weights : numpy array
        1D array of SEM weights
    lQ1 : numpy array
        1D array of Eigenvalues of H0 operator for the first dynamics.
    Qx1 : numpy array
        2D array of Eigenfunctions of H0 operator for the first dynamics.
    lQ2 : numpy array
        1D array of Eigenvalues of H0 operator for the second dynamics.
    Qx2 : numpy array
        2D array of Eigenfunctions of H0 operator for the second dynamics.
    terminal_time : float, optional
        Terminal time that defines upper limit of the time integral. The default is 0.5.
    number_of_samples : int, optional
        Number of time steps for numerical approxination of the time integral. The default is 5.

    Returns
    -------
    out : float
        JS divergence between the two time-dependent distributions.
    """
    time = np.linspace(0, terminal_time, number_of_samples)
    out = JS_divergence(p01, p02, weights) / 2
    rho0d1 = Qx1.T.dot(np.diag(weights)).dot(p01 / np.sqrt(peq1))
    rho0d2 = Qx2.T.dot(np.diag(weights)).dot(p02 / np.sqrt(peq2))

    for i in range(1, len(time)):
        p1 = (Qx1.dot(rho0d1.dot(
            np.diag(np.exp(-lQ1 * (time[i] - time[0])))))) * np.sqrt(peq1)
        p2 = (Qx2.dot(rho0d2.dot(
            np.diag(np.exp(-lQ2 * (time[i] - time[0])))))) * np.sqrt(peq2)
        out += JS_divergence(p1, p2, weights, 'unnormalized')
        # print(out)
    return out * (time[1] - time[0])


