# -*- coding: utf-8 -*-

# This is a part of EnergyModel class
# This source file contains public functions related to fitting, result saving, and switching between F and peq representations

import numpy as np
import os.path

def score(self, data, peq=None, D=None, fr=None):
    """Evaluate negative log-likelihood for a given data and model
    
    
    Parameters
    ----------
    data : numpy array (N,2),dtype=np.ndarray. 
        Spike data packed as numpy array of the size (N,2), where each elements is a 1D array of floats. 
        N is the number of trials, and for each trial the first column contains interspike intervals (ISIs), 
        and the second column contains the corresponding neuronal indices (in the paper only one neural 
        response was always used, so the neuronal indices are all set to 0). ISIs are represented as 1D arrays
        of floats, and neuronal indices as 1D array of integers.
    peq : numpy array, dtype=float
        The equilibrium probability density evaluated on SEM grid. The default is self.peq_    
    D : float
        Noise intensity. The default is self.D_ 
    fr : numpy array, dtype=float
        2D array that contains firing rate functions for each neuron evaluated on SEM grid. The default is self.fr_
        
    Returns
    -------
    likelihood : double
        negative loglikelihood of a data given a model
    """
    # Run data checks
    self._check_data(data)
    
    if peq is None:
        peq=self.peq_
    if fr is None:
        fr=self.fr_
    if D is None:
        D=self.D_
        
    if self.verbose:
        print("Evaluating model likelihood on the provided data...")
    
    return self._get_loglik_data(data, peq, D, fr, 'likelihood')['loglik']
    
def fit(self, data, optimizer='GD', options=None, save_iter=None):
    """Perform model fitting 
    
    
    Parameters
    ----------
    data : numpy array (N,2),dtype=np.ndarray. 
        Spike data packed as numpy array of the size (N,2), where each element is a 1D array of floats. 
        N is the number of trials, and for each trial the first column contains interspike intervals (ISIs), 
        and the second column contains the corresponding neuronal indices (in the paper only one neural 
        response was always used, so the neuronal indices are all set to 0). ISIs are represented as 1D arrays
        of floats, and neuronal indices as a 1D array of integers.
        
    optimizer : str
        choose optimization strategy. The default is 'GD' 
    
    options : 
        see self._GD_optimization for availiable options of GD optimizer. The default is None
    
    save_iter : dictionary 
        dictionary with 'path' and 'stride'[=max_possible] entries for the intermediate file saving
        Example: save_iter={'path': 'my_simulations', 'stride': 10} will save the results in 
        'my_simulation' folder every 10 iterations. The folder should be created in advance. The default is None.
    
    Returns
    -------
    em : self
        A fitted EnergyModel object.
    """
    
    # Run input checks
    self._check_data(data)
    
    # Check optimizer
    if optimizer not in self._optimizer_types:
        raise NotImplementedError("optimizer should be one of %s"
                         % self._optimizer_types)
    
    if options is None:
        options = {}
        
    # reset self.converged_ to False
    self.converged_ = False
    
    # Maximum Likelihood Estimation of the parameters
    if self.verbose:
        print("Performing Estimation of the model parameters...")
        print("The chosen optimizer is: " + str(optimizer))
    
    # Save initial parameters:
    if not hasattr(self,'peq_init'):
        self.peq_init=np.copy(self.peq_)
    if not hasattr(self,'fr_init'):
        self.fr_init=np.copy(self.fr_)
    if not hasattr(self,'D_init'):
        self.D_init=np.copy(self.D_)
    
    if optimizer == 'GD':
        self._check_optimization_options(optimizer, options, save_iter)
        self.iterations_GD_ = self._GD_optimization(data, save_iter, **options)
    else:
        raise NotImplementedError("This optimizer {} is not implemented yet".format(optimizer))
    return self

def SaveResults(self, results_type='iterations', **options):
    """Save fitting results to a file
    
    
    Parameters
    ----------
    results_type : str 
        Type of the results to save. Supported types: 'iterations', which saves iteration results. The default is 'iterations'
    options : dict
        Availiable options:
            results : dictionary 
                dictionary with the results (mandatory)
            path : str 
                path to files. The default is empty str
            name : str 
                enforce a particular file name, otherwise, the default name will be generated.
            sim_id : int
                id of a simulation (appended to the name and saved inside the dictionary). The default is empty str.
            iter_start : int
                Starting iteration used for automatic filename generation. The default is iter_num[0]
            iter_end : int
                Terminate iteration used for automatic filename generation. The default is iter_num[-1]
    """
    
    #Insert '/' in path string if necessary
    if 'path' in options:
        if options['path'][-1]!='/' and len(options['path'])>0:
            options['path']+='/'
        path=options['path']
    else:
        path=''
    
    #Exctract simulation ID if provided
    if 'sim_id' in options:
        sim_id=options['sim_id']
        if sim_id=='':
            print("Warning: sim_id not provided")    
    else:
        print("Warning: sim_id not provided")
        sim_id=''
        
    #for each result_type defate name-prefix and data dictionary 
    if results_type == 'iterations':
        prefix='results'
        if 'results' in options:
            dictionary=options['results']
        else:
            raise ValueError("Please specify data_dictionary")
    else:
        raise ValueError(results_type + " is not supported")
        
    #Add sim_id if providied:
    if isinstance(sim_id,int):
        dictionary['sim_id']=sim_id
    
    #Generate name and save
    if 'name' in options:
        fullname=path+options['name']
    else:
        if results_type=='iterations':
            if 'iter_start' in options:
                iter_start=str(options['iter_start'])
            elif 'iter_num' in dictionary:
                iter_start=str(dictionary['iter_num'][0])
            else:
                iter_start='0' 
            iter_start+='-'
            if 'iter_end' in options:
                iter_end=str(options['iter_end'])
            elif 'iter_num' in dictionary:
                iter_end=str(dictionary['iter_num'][-1])
            else:
                iter_end=str(self.iterations_GD_['logliks'].size) 
            postfix='_iterations_'
        fullname=path +prefix+str(sim_id)+ postfix+ iter_start+iter_end+'.npz'
    if os.path.isfile(fullname):
        print('Error: file '+fullname+' already exists. Aborting...')
        return
    else:
        np.savez(fullname,**dictionary)
        print ('file '+fullname+ ' saved.')


    
def calc_peq(self, F):
    """Calculate peq from the force
    
    Calculates equilibrium probability distribution from the provided driving force
    

    Parameters
    ----------
    F : numpy array, dtype=float
        Driving force, 1D array

    Returns
    -------
    result : numpy array, dtype=float
        Equilibirum probability distribution (peq), 1D array

    """
    result=np.exp(self.Integrate(F))
    result/=np.sum(result*self.w_d_)
    return result

def calc_F(self, peq):
   """Calculate force from the peq
   
   Calculates driving force from the provided equilibrium probability distribution
    

    Parameters
    ----------
    peq : numpy array, dtype=float
        Equilibirum probability distribution (peq), 1D array

    Returns
    -------
    F : numpy array, dtype=float
        Driving force, 1D array
    """
   return self.dmat_d_.dot(np.log(peq))





