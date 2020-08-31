# -*- coding: utf-8 -*-

# This is a part of EnergyModel class
# This source file contains functions related to gradient descent optimization
import numpy as np
import sys

def FeatureComplexity(self,peq=None):
    """Calculate feature complexity from a givein peq
    

    Parameters
    ----------
    peq : numpy array, dtype=float
        The equilibrium probability density evaluated on SEM grid. The default is self.peq_   

    Returns
    -------
    float
        Feature complexity

    """
    
    if peq is None:
        peq=self.peq_
    return np.sum(4*((self.dmat_d_.dot(np.sqrt(peq)))**2)*self.w_d_)

def FeatureComplexityFderiv(self,peq=None):
    """Variational derivative of feature complexity wrt force F
    

    Parameters
    ----------
    peq : numpy array, dtype=float
        The equilibrium probability density evaluated on SEM grid. The default is self.peq_ 

    Returns
    -------
    numpy array
        The variational derivative of loglikelihood w.r.t. feature complexity.

    """
    if peq is None:
        peq=self.peq_
    return self.Integrate(-4*((self.dmat_d_.dot(np.sqrt(peq)))**2)+2*self.dmat_d_.dot(self.dmat_d_.dot(peq))+self.FeatureComplexity(peq=peq)*peq)


def _GD_optimization(self, data, save_iter, gamma, max_iteration, loglik_tol, sim_start, 
                     schedule, dataCV, etaf, sim_id):
        """Perfrom gradient-descent optimization

        Parameters
        ----------
        data : numpy array (N,2),dtype=np.ndarray. 
            Spike data packed as numpy array of the size (N,2), where each elements is a 1D array of floats. 
            N is the number of trials, and for each trial the first column contains inter spike intervals (ISIs), 
            and the second column contains the corresponding neuronal indices (in the paper only one neural 
            response was always used, so the neuronal indices are all set to 0). ISIs are represented as 1D arrays
            of floats, and neuronal indices as 1D array of integers.
            data[any Trial][0] - 1D array, ISIs of type float64
            data[any Trial][1] - 1D array, neuronal IDs of type int64
        save_iter : dictionary 
            Dictionary with 'path' and 'stride'[=max_possible] entries for the intermediate file saving.
            If not provided, results will not be saved.
            Example: save_iter={'path': 'my_simulations', 'stride': 10} 
            will save the results in 'my_simulation' folder every 10 iterations. The default is None.           
        gamma : dicrionary 
            Dictionary that specifies the learning rates. Also specifies which parameters are to be optimized. Availible options are 'F' and 'D'
            Example: gamma={'F':0.01} - will optimize the driving force only, gamma={'F':0.01, 'D':0.001} will optimize both the driving force and 
            noise magnitude. 
        max_iteration : int 
            Maximum number of iterations. The default is 100
        loglik_tol : float 
            Tolerance for the optimization termination. If relative loglikelihood improvement is less than that, stop the
            optimization. If set to zero, the optimization will not be stopped due to finite machine precision (it this case the optimization
            will terminate after reaching max_iteration). The default is 0.    
        sim_start : int
            starting iteration number (0 if new simulation, otherwise int >0). The default is 0 
        schedule : numpy array, dtype=int
            1D array with iteration numbers on which the fitted parameters (e.g. peq and/or D)
            are saved to RAM/file. The default is np.arange(1-new_sim, max_iteration+1), which includes all 
            the iteration numbers, and saves fitting results on each iteration 
        dataCV : numpy array (N_cv,2), dtype=np.ndarray.
            Validation data on which negative validated likelihood will be computed on every iteration. The format is the same as for training
            data, and N_cv is number of validation trials.
            The default is None, in which case no validation will be performed.
        etaf: float 
            Regularization strength (set to zero for unregularized optimization). The default is 0.
        sim_id: int
            A unique simulation id needed to generate the filename for the files with the results. The default is None 
        
        Returns
        -------
        results: dictionary 
            Dictionary with the fitted parameters. The possible entries are:
            'loglik': training loglikelihood saved on each iteration.
            'loglikCV': validation loglikelihood saved on each iteration (only if dataCV is provided).
            'peqs': fitted equilibirum probability distributions saved on the iterations specified by the schedule array (only if 'F' is in gamma). 
            'Ds': fitted diffusion coefficients saved on the iterations specified by the schedule array  (only if 'D' is in gamma). 
        """
        # initialize parameters
        peq = self.peq_
        D = self.D_
        fr=self.fr_
        
        #Determine if it is a new simulation:
        new_sim=1 if sim_start==0 else 0
        
        #Starting indexes in results_save dict for the parameters and logliks
        iter_start=0
        iter_start_ll=0
                
        #Index (in the schedule array) of the next iteration to be saved 
        next_save=new_sim
        
        #Define list of parameters to be optimized
        params_to_opt=list(gamma.keys())
        
        #Determine if we have validation data
        CV=True if dataCV is not None else False
        
        #Initialize results dictionary, save initial parameters 
        results={}
        results['logliks'] = np.zeros(max_iteration+new_sim)
        if CV:
            results['logliksCV']=np.zeros(max_iteration+new_sim)  
        results['iter_num']=np.zeros(schedule.size, dtype='int64')
        if new_sim:
            results['iter_num'][0]=0
        if  'peq' in gamma.keys() or 'F' in gamma.keys():
            results['peqs'] = np.zeros((len(peq), schedule.size))   
            if new_sim:
                results['peqs'][...,0]=peq
        if 'D' in params_to_opt:
            results['Ds'] = np.zeros(schedule.size)
            if new_sim:
                results['Ds'][0]=D
                
        # Run GD
        oldLL = 0

        for i,iIter in enumerate(range(new_sim+sim_start,new_sim+sim_start+max_iteration)):
            if self.verbose:
                print('Iteration {}'.format(iIter))
                        
            # compute gradients 
            gradients = self._get_loglik_data(data, peq, D, fr, 'gradient', params_to_opt)
            
            #extract loglikelihoods
            ll=gradients['loglik']
            if CV:
                #validation score
                llCV=self.score(dataCV,peq,D,fr)
            
            #Update the parameters
            if 'F' in params_to_opt:
                #find the current force, update it and calculate peq from it
                F = (self.dmat_d_.dot(np.log(peq)))
                F -= gamma['F']*gradients['F']
                if etaf!=0:
                    F -= etaf*self.FeatureComplexityFderiv(peq=peq)
                peq =np.exp(self.Integrate(F))
                peq=np.maximum(peq,10**(-10)) #For numerical stability
                peq/=np.sum(peq*self.w_d_)
            if 'D' in params_to_opt:
                D-=gamma['D']*gradients['D'] 
                D=np.maximum(D,0)
            
            
            #Update loglik (delayed  by 1 iteration)
            if i+new_sim>0:
                results['logliks'][i+new_sim-1] = ll
                if CV:
                    results['logliksCV'][i+new_sim-1]=llCV
            
            #Save iteration to the results dict 
            if iIter==schedule[next_save]:               
                if 'F' in params_to_opt or 'peq' in params_to_opt:
                    results['peqs'][...,next_save]=peq
                if 'D' in params_to_opt:
                    results['Ds'][...,next_save]=D
                results['iter_num'][next_save]=iIter
                next_save+=1
            
            #Write intermediate results in to a file
            if save_iter is not None and (i+1)%save_iter['stride']==0:
                #Create and fill intermediate results_save dictionary
                results_save={}
                for keys in results:
                    if keys.startswith('log'):
                        results_save[keys]=results[keys][...,iter_start_ll:i+1+new_sim]
                    else:
                        results_save[keys]=results[keys][...,iter_start:next_save]
                
                #Score the last iteration
                results_save['logliks'][-1]=self.score(data, peq, D, fr)
                if CV:
                    results_save['logliksCV'][-1]=self.score(dataCV, peq, D, fr)    
            
                #Save to file
                self.SaveResults('iterations', optimizer='GD', path=save_iter['path'], 
                                 iter_start=schedule[iter_start], iter_end=iIter, results=results_save, sim_id=sim_id)
                
                #Update saving indices
                iter_start=next_save
                iter_start_ll=i+1+new_sim
                
            # check for convergence
            if abs(ll - oldLL) / (1 + abs(oldLL)) < loglik_tol:
                if self.verbose:
                    print('Optimization converged in {} iterations'.format(iIter + 1))
                self.converged_ = True
                break

            oldLL = ll
            
            # flush std out
            sys.stdout.flush()
        
        
            
        if (self.verbose & (i+1 == (max_iteration - 1))):
            print("Warning: convergence has not been achieved "
                  "in {} iterations".format(max_iteration+1))
        
        #Adjust results if algorithm convereged earlier. Update parameters
        all_iter = slice(next_save)
        if  'peq' in params_to_opt or 'F' in params_to_opt:
            results['peqs'] = results['peqs'][...,all_iter]
            self.peq_=np.copy(results['peqs'][...,-1])
        if 'D' in params_to_opt:
            results['Ds'] = results['Ds'][all_iter]
            self.D_=np.copy(results['Ds'][...,-1])
        
        results['iter_num']=results['iter_num'][all_iter]
        results['logliks'] = results['logliks'][0:i+1+new_sim]
        if CV:
            results['logliksCV'] = results['logliksCV'][0:i+1+new_sim]
        
        #Calculate the last loglik
        results['logliks'][i+new_sim]=self.score(data, peq, D, fr)
        if CV:
            results['logliksCV'][i+new_sim]=self.score(dataCV, peq, D, fr)
        
        #Save the final chunk of data
        if save_iter is not None and iter_start<next_save:
            results_save={}
            for keys in results:
                if keys.startswith('log'):
                    results_save[keys]=results[keys][...,iter_start_ll:i+1+new_sim]
                else:
                    results_save[keys]=results[keys][...,iter_start:next_save]
            self.SaveResults('iterations', optimizer='GD', path=save_iter['path'], 
                        iter_start=schedule[iter_start], iter_end=iIter, results=results_save, sim_id=sim_id)
        
        return results
        

