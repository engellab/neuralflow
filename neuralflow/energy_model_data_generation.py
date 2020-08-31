# -*- coding: utf-8 -*-

# This is a part of EnergyModel class
# This source file contains functions for synthetic data-generation 

import numpy as np
import math
from tqdm import tqdm
from .energy_model_settings import MINIMUM_PEQ

def generate_data(self, deltaT=0.00001, time_epoch = [(0,1)]):
    """Generate spike data and latent trajectories 
    
    
    Parameters
    ----------
    deltaT : float
        Size of the time bin in seconds for the integration of the Langevin equation. The default is 0.00001.
    time_epoch : list of tuples
        List of N tuples, where N is the number of trials. Each tuple consists of start time and stop time in seconds. The default is [(0,1)].
 
    Returns
    -------
    data : numpy array (N,2),dtype=np.ndarray. 
        Spike data packed as numpy array of the size (N,2), where each elements is a 1D array of floats. 
        N is the number of trials, and for each trial the first column contains inter spike intervals (ISIs), 
        and the second column contains the corresponding neuronal indices (in the paper only one neural 
        response was always used, so the neuronal indices are all set to 0). ISIs are represented as 1D arrays
    
    time_bins : numpy array (N,), dtype = np.ndarray 
        For each trial contains times at which latent trajectory was recorded. N is the number of trials,
        and for each trial time is represented as 1D array of floats.  
        
    x : numpy array (N,), dtype = np.ndarray 
        Latent trajectories for each trial, N is the number of trials. Each entry is 1D array of floats.  
        
    """
    return self._generate_data(self.peq_, self.D_, self.firing_model_, self.num_neuron, deltaT, time_epoch)
    
def _generate_data(self, peq, D, firing_rate_model, num_neuron, deltaT, time_epoch):
    """Generates synthetic spike data and latent trajectories from a given model defined by (peq,D,firing_rate_model). 
    
    
    Parameters
    ----------
    peq : numpy array, dtype=float 
        Equilibirum probabilioty distribution that determines potential Phi(x), see Suplementary Note 1.1. 1D array.     
    D : float
        Noise magnitude.     
    firing_rate_model : list
        List of lambda functions to calculate firing rates, one for each neuron.  
    num_neuron : int
        Number of neural responces.     
    deltaT : float
        Size of time bin in seconds used for integration of Langevin equation.     
    time_epoch : list of tuples
         List of N tuples, where N is the number of trials. Each tuple consists of start time and stop time in seconds.
         
    Returns
    -------
    data : numpy array (N,2),dtype=np.ndarray. 
        Spike data packed as numpy array of the size (N,2), where each elements is a 1D array of floats. 
        N is the number of trials, and for each trial the first column contains interspike intervals (ISIs), 
        and the second column contains the corresponding neuronal indices (in the paper only one neural 
        response was always used, so the neuronal indices are all set to 0). ISIs are represented as 1D arrays
    time_bins : numpy array (N,), dtype = np.ndarray 
        For each trial contains times at which latent trajectory was recorded. N is the number of trials,
        and for each trial time is represented as 1D array of float.       
    x : numpy array (N,), dtype = np.ndarray 
        Latent trajectories for each trial, N is the number of trials. Each entry is 1D array of floats.  
        
    """
    num_trial = len(time_epoch)
    
    # generate diffusion trajectories
    x, time_bins = self._generate_diffusion(peq, D, deltaT, time_epoch)
    
    # initialize data arrays
    rate = np.empty((num_neuron,num_trial),dtype=np.ndarray)
    spikes = np.empty((num_neuron,num_trial),dtype=np.ndarray)
    
    # generate firing rates and spikes
    for iTrial in range(num_trial):
        for iCell in range(num_neuron):
            #Firing rate f(x(t))
            rate[iCell, iTrial] = firing_rate_model[iCell](x[iTrial])
            rt = rate[iCell,iTrial]
            #Generate spikes from rate
            spikes[iCell, iTrial] = self._generate_inhom_poisson(time_bins[iTrial][0:rt.shape[0]], rate[iCell,iTrial])
        
    # transform spikes to ISIs
    data = self.transform_spikes_to_isi(spikes, time_epoch)
        
    return data, time_bins, x

def transform_spikes_to_isi(self, spikes, time_epoch):
    """Convert spike times to data array 


    Parameters
    ----------
    spikes : numpy array (N,), dtype=np.ndarray 
        A sequence of spike times for each of the trials, N is the number of trials. Each entry is 1D array of floats  
    time_epoch : list of tuples
         List of N tuples, where N is the number of trials. Each tuple consists of the trial's start time and stop time in seconds.

    Returns
    -------
    data : numpy array (N,2),dtype=np.ndarray. 
        Spike data packed as numpy array of the size (N,2), where each elements is a 1D array of floats. 
        N is the number of trials, and for each trial the first column contains interspike intervals (ISIs), 
        and the second column contains the corresponding neuronal indices (in the paper only one neural 
        response was always used, so the neuronal indices are all set to 0). ISIs are represented as 1D arrays.
        
    """
    num_neuron, num_trial = spikes.shape
    # initialize data array
    data = np.empty((num_trial,2),dtype=np.ndarray)
    
    # indices of neurons that spiked
    spike_ind = np.empty(num_neuron, dtype=np.ndarray)
    
    # transform spikes to interspike intervals format
    for iTrial in range(num_trial):
        for iCell in range(num_neuron):
            spike_ind[iCell] = iCell*np.ones(len(spikes[iCell, iTrial]), dtype=np.int)       
        all_spikes = np.concatenate(spikes[:,iTrial],axis=0)
        all_spike_ind = np.concatenate(spike_ind[:],axis=0)
        if all_spikes.shape[0] == 0:
            data[iTrial, 0] = np.zeros(0)
            data[iTrial, 1] = np.zeros(0)
            
        # sort spike times and neuron index arrays
        ind_sort = np.argsort(all_spikes)
        all_spikes = all_spikes[ind_sort]
        all_spike_ind = all_spike_ind[ind_sort]
        # compute interspike intervals
        data[iTrial,0] = np.zeros(len(all_spikes), dtype=np.double)
        data[iTrial,0][1:] = all_spikes[1:] - all_spikes[:-1]
        data[iTrial,0][0] = all_spikes[0] - time_epoch[iTrial][0] # handle the first ISI
        # assign indicies of neurons which fired
        data[iTrial,1] = all_spike_ind
        
    return data
     
def _generate_inhom_poisson(self, time, rate):
    """Generate spike sequence from a given rate of inhomogenious Poisson process lambda(t) and time t
    
    
    Parameters
    ----------
    time : numpy array, dtype=float
        1D array of all time points 
    rate : numpy array, dtype=float
        1D array of the corresponding firing rates 
        
    Returns
    -------
    spikes : np.array,  dtype=float
        1D array of spike times

    """
    # calculate cumulative rate
    deltaT = time[1:]-time[:-1]
    r = np.cumsum( rate[0:-1]*deltaT )
    r = np.insert(r, 0, 0)
    deltaR = r[1:] - r[:-1]
    
    # generate 1.5 as many spikes as expected on average for exponential distribution with rate 1
    numX = math.ceil( 1.5*r[-1] )
    
    # generate exponential distributed spikes with the average rate 1
    notEnough = True
    x = np.empty(0)
    xend = 0.0
    while notEnough:
        x = np.append(x, xend+np.cumsum( np.random.exponential(1.0, numX) ) )
        # check that we generated enough spikes
        if (not len(x)==0):
            xend = x[-1]
        notEnough = xend<r[-1]

    # trim extra spikes
    x = x[ x<=r[-1] ]
    
    if len(x)==0:
        spikes = np.empty(0)
    else:
        # for each x find index of the last rate which is smaller than x
        indJ = [np.where( r <= x[iSpike] )[0][-1] for iSpike in range(len(x)) ]

        # compute rescaled spike times
        spikes = time[indJ] + (x-r[indJ])*deltaT[indJ]/deltaR[indJ]

    return spikes
        
        
def _generate_diffusion(self, peq, D, deltaT=0.00001, time_epoch = [(0,1)]):
    """Sample latent trajectory by integration of Langevin equation
    

    Parameters
    ----------
    peq : numpy array, dtype=float 
        Equilibirum probabilioty distribution that determines potential Phi(x), see Suplementary Note 1.1. 1D array.     
    D : float
        Noise magnitude.     
    deltaT : float
        Size of time bin in seconds used for integration of Langevin equation. The default is 0.00001.
    time_epoch : list of tuples
         List of N tuples, where N is number of trials. Each tuple consists of start time and stop time in seconds.The default is [(0,1)].

    Returns
    -------
    x : numpy array, dtype=float
        Latent trajectory, 1D array of floats
    time_bins : numpy array, dtype=float
       Times at which latent trajectory was recorded, 1D array of floats
        
    """
        
    num_trial = len(time_epoch)
    # pre-allocate output
    x = np.empty(num_trial,dtype=np.ndarray)
    time_bins = np.empty(num_trial,dtype=np.ndarray)

    #TODO: rewrite initial conditions so that it samples enough from within the boundaries

    # sample initial condition from the equilibrium distribution
    x0 = self._sample_from_p(peq, num_trial)

    # Normalization
    p0 = np.maximum(peq, 0)
    p0 += MINIMUM_PEQ
    p0 /= self.w_d_.dot(p0)
    # compute force profile from the potential
    force = (self.dmat_d_.dot(p0))/p0
    len_xd = len(force)

    if self.verbose:
            iter_list = tqdm(range(num_trial))
    else:
            iter_list = range(num_trial)
    
    for iTrial in iter_list:
        # generate time bins
        time_bins[iTrial] =  np.arange(time_epoch[iTrial][0],time_epoch[iTrial][1],deltaT)
        num_bin = len( time_bins[iTrial] )
        y = np.zeros(num_bin+1)
        y[0] = x0[iTrial]
        
        # generate random numbers
        noise = np.sqrt(deltaT*2*D)*np.random.randn(num_bin)
        
        #account for absorbing boundary trajectories ending early
        max_ind = num_bin+1

        # do Euler integration
        for iBin in range(num_bin):
            # find force at the current position
            ind = np.argmax( self.x_d_ - y[iBin] >= 0 )
            if ind==0:
                f = force[0] 
            elif ind==(len_xd-1):
                f = force[-1]
            else:
                theta = (y[iBin] - self.x_d_[ind-1])/(self.x_d_[ind]-self.x_d_[ind-1])
                f = (1.0-theta)*force[ind-1] + theta*force[ind]
            
            y[iBin+1] = y[iBin] + D*f*deltaT + noise[iBin]
            
            # Handle boundaries:
            y[iBin+1]=min(max(y[iBin+1], 2*self.x_d_[0]-y[iBin+1]), 2*self.x_d_[-1]-y[iBin+1])
            
        x[iTrial] = y[1:max_ind]
    
    return x, time_bins
    
def _sample_from_p(self, p, num_sample):
    """Generate samples from a given probability distribution. Needed for initialization of the latent trajectories
    
    
    Parameters
    ----------
    p : numpy array, dtype=float
        The probability distribution, 1D array of floats
    num_sample : int
        Number of samples

    Returns
    -------
    x : numpy array, dtype=float
        1D array of size num_sample that consists of samples randomnly drawn from a given probaiblity distribution

    """
    x = np.zeros(num_sample)
    pcum = np.cumsum(p*self.w_d_)

    y = np.random.uniform(0,1,num_sample)

    for iSample in range(num_sample):
        # find index of the element closest to y[iSample]
        ind = (np.abs(pcum-y[iSample])).argmin()

        x[iSample] = self.x_d_[ ind ]
    
    return x
