#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
feature_consistency module
--------------------------------------------------

Below are the complimentary functions for Model selection based on feature complexity analysis.
This module is optional and should be imported separately: from neuralflow.utilities import feature_consistency
"""
import numpy as np

def FeatureConsistencyAnalysis(data1,data2, grid_params, KL_thres, FC_radius,  FC_final, FC_stride):
    """ Perform feature consistency analysis on two fitting results obtained from different data samples. 
    The algorithm works as follows:
        1) Create a shared feature complexity axis.
        2) For each entry in shared FC, select all models from data1 and data2 that differ from the current FC by less than FC_radius.
        3) For all selected models, calculate each of the pairwise KL divergences. 
        4) Select and save minimal KL divergence and the corresponding indices in the original data1/data2 arrays.
    
    
    Parameters
    ----------
    data1 : dictionary
        Dictionary with two entries: 'peqs' and 'FCs' for fitting results on data sample 1. 
        The first entry contains 2D array of peqs, and the second - 1D array of the corresponding feature complexities.
    data2 : dictionary
        Dictionary with two entries: 'peqs' and 'FCs' for fitting results on data sample 2. 
        The first entry contains 2D array of peqs, and the second - 1D array of the corresponding feature complexities.
    grid_params : dictionary
        Contains two entries: 'w': SEM weights and 'dmat': SEM differentiation matrix. 
        Can be extracted from Energymodel variable: grid_params = {'w': EnergyModel..w_d_, 'dmat': EnergyModel..dmat_d_}
    KL_thres : float
        Threshold KL divergence for determining optimal feature complexity. 
    FC_radius : float
        Feature complexity radius that determines maximum slack in features complexities of two models.
    FC_final : float
        Maximum feature complexity on shared feature complexity axis
    FC_stride : float
        Feature complexity resolution on shared feature complexity axis

    Returns
    -------
    FC_shared : numpy array, dtype=float
        1D array of shared feature complexities (shared feature complexity axis)
    KL : numpy array, dtype=float
        1D array of KL divergences between the two models.
    FC_opt_ind : int
        Index of optimal KL divergence in FC_shared array
    ind1_shared : numpy array, dtype=int
        Indices of peqs/FCs in data1 array that correspond to each entry in FC_shared. 
    ind2_shared : numpy array, dtype=int
        Indices of peqs/FCs in data2 array that correspond to each entry in FC_shared. 

    """
    #Extract FCs and peqs for datasample 1 and 2
    FCs1 = data1['FCs']
    FCs2 = data2['FCs']
    
    peqs1=data1['peqs']
    peqs2=data2['peqs']

    #Shared feature complexity axis
    FC_all=np.linspace(0,FC_final,int(FC_final/FC_stride+1))
    
    #Initialize index arrays 
    ind1=np.zeros(FC_all.size) 
    ind2=np.zeros(FC_all.size)
    
    FC_all_reshaped = np.repeat(FC_all[np.newaxis,:],FCs1.size,axis=0)
    
    #Find indices of peqs/FCs in data1 and data2 array that correspond to each entry of FC_all
    iter_FC_all1 = np.argmin(np.abs(FC_all_reshaped - np.repeat(FCs1[:,np.newaxis],FC_all.size,axis=1)),axis=0) 
    iter_FC_all2 = np.argmin(np.abs(FC_all_reshaped - np.repeat(FCs2[:,np.newaxis],FC_all.size,axis=1)),axis=0) 
    
    KL = np.zeros(FC_all.shape)
    
    for i in range(len(FC_all)):
        #For each level of FC, extract peqs1 and peqs2 that differ from FC by less than FC_radius
        peqs_FC_1 = peqs1[:,iter_FC_all1[np.maximum(0,i-int(FC_radius/FC_stride)):np.minimum(FC_all.size,i+int(FC_radius/FC_stride)+1)]]
        peqs_FC_2 = peqs2[:,iter_FC_all2[np.maximum(0,i-int(FC_radius/FC_stride)):np.minimum(FC_all.size,i+int(FC_radius/FC_stride)+1)]]
        
        if peqs_FC_1.shape[1]==0 or peqs_FC_2.shape[1]==0:
            KL[i] = 0
        else:
            #Caculate all pairwise symmetric KLs and select the minimal one.
            a=0.5 * (KL_divergence(peqs_FC_1,peqs_FC_2, grid_params['w'],'all').T+KL_divergence(peqs_FC_2,peqs_FC_1,grid_params['w'],'all'))
            ind1[i]= np.where(a == a.min())[0][0] + np.maximum(0,i-int(FC_radius/FC_stride))
            ind2[i]= np.where(a == a.min())[1][0] + np.maximum(0,i-int(FC_radius/FC_stride))
            KL[i] = np.min (a)
    ind1_shared = iter_FC_all1[ind1.astype(int)]
    ind2_shared = iter_FC_all2[ind2.astype(int)]
    
    FC_shared = 0.5* (FCs1[ind1_shared]+ FCs2[ind2_shared])
    
    #Find an index of optimal FC
    FC_opt_ind = np.where(KL > KL_thres)[0][0]-1
    
    return FC_shared, KL, FC_opt_ind, ind1_shared, ind2_shared


def FeatureComplexityBatch(peqs, grid_params, start_index=0, stop_index=-1):
    """Calculate Feature complexities for a given array of peqs
    

    Parameters
    ----------
    peqs : numpy array
        2D array of peqs of size (N1,N2), where N1 - number of grid points, N2 - number of peqs.
    grid_params : dictionary
        Contains two entries: 'w': SEM weights and 'dmat': SEM differentiation matrix. 
        Can be extracted from Energymodel variable: grid_params = {'w': EnergyModel.w_d_, 'dmat': EnergyModel.dmat_d_}
    start_index : int
        Starting index that determines the number of points at the left boundary to be skipped for FC calculation. The default is 0.
    stop_index : int
        Stop index that determines the number of points at the right boundary to be skipped for FC calculation. The default is -1.

    Returns
    -------
    FCs: numpy array
        1D array of shape (N2,) that contains FC for each of the provided peqs

    """
    if len(peqs.shape)==1:
        peqs=peqs[:,np.newaxis]
    N2 = peqs.shape[1] 
    FCs=np.zeros(N2)
    for i in range(N2):
        rh=np.sqrt(peqs[...,i])
        FCs[i]=np.sum(((grid_params['dmat'].dot(rh))**2)[start_index:stop_index]*grid_params['w'][start_index:stop_index])
    return FCs
        
def KL_divergence(peq1,peq2,weights,schema='pairwise'):
    """Calculate KL divergence between two batches of distributions peq1 and peq2. 
    KL = integral (p1*log(p1/p2))
    
    
    Parameters
    ----------
    peq1: numpy array
        2D array of the first batch of distributions (each column contains a distribution)
    peq2: numpy array 
        2D array of the second batch of distributions (each column contains a distribution)
    weights: numpy array
        1D array of SEM weights 
    schema: ENUM('pairwise','all') 
       'pairwise' mode calculates KL between each pair (peq1[:,0] peq2[:,0]), (peq1[:,1] peq2[:,1]), and so on
       'all' mode calculates KL between all possible pairs from peq1 and peq2, e.g. 
        if peq1 contains 3 models and peq2 contains 4 models, 12 KL divergences will be calculated and packed in a 3x4 matrix.
        
    Returns
    -------
    numpy array
        Array with KL divergences
     
    """
    if len(peq1.shape)==1 and len(peq2.shape)==1: 
        return np.sum(weights*peq1*np.log(peq1/peq2))
    elif schema=='pairwise':
        assert peq1.shape==peq2.shape, "peq1 and peq2 must be of equal shape for the pairwise mode"
        return np.sum(weights*(peq1*np.log(peq1/peq2)).T,axis=1)
    elif schema=='all':
        return np.diagonal(np.tensordot((weights*peq1.T),np.log(peq1[...,None]/peq2[:,None]),axes=1))