# -*- coding: utf-8 -*-

# This is a part of EnergyModel class
# This source file contains template firing rate functions 



import numpy as np


def custom(x,lambdafunc=None):
    """Custom fr model.
    Either supply a function, or leave None if the model of the generated data is not known
    

    Parameters
    ----------
    x : numpy array, dtype=float
        SEM grid points
    lambdafunc : function.
        The default is None.

    Returns
    -------
    fr : numpy array
        Firing rate function evaluated on SEM grid

    """
    if lambdafunc is not None:
        fr=lambdafunc(x)
    else:
        fr=None
    return fr

def rectified_linear(x, r_slope=50.0, x_thresh=-1.0):
    """Rectified-linear firing rate model.
    x, r_slope, x_thresh --> r(x, r_slope, x_thresh) = max[ r_slope*(x - x_thresh), 0 ]

    Parameters
    ----------
    x : numpy array, dtype=float
        SEM grid points
    r_slope : float
        Firing-rate slope parameter. The default is 50.0.        
    x_threshold : float
        Firing threshold parameter. The default is -1.0.

    Returns
    -------
    numpy array 
        Firing rate function evaluated on SEM grid
    """
    
    return np.maximum(r_slope*(x-x_thresh),0.0)

def linear (x, r_slope = 50.0, r_bias = 2):
    """Linear firing rate model.
    

    Parameters
    ----------
    x : numpy array, dtype=float
        SEM grid points
    r_slope : float
        Firing-rate slope parameter. The default is 50.0.
    r_bias : float
        Firing threshold parameter. The default is 2.

    Returns
    -------
    numpy array 
        Firing rate function evaluated on SEM grid

    """
    return np.maximum(r_slope*x+r_bias,0.0)

def peaks (x, center=np.array([-0.5,0.5]), width=np.array([0.2,0.3]), amp=np.array([1000,800])):
    """Sum of gaussian peaks
    f= SUM( A*exp((x-x0)^2/w^2) )
    
    Parameters
    ----------
    x : numpy array, dtype=float
        SEM grid points
    center : numpy array, dtype=float
        Centers of Gaussian peaks. The default is np.array([-0.5,0.5]).
    width : numpy array, dtype=float
        Widths of Gaussian peaks. The default is np.array([0.2,0.3]).
    amp : numpy array, dtype=float
        Magnitudes of Gaussian peaks. The default is np.array([1000,800]).

    Returns
    -------
    numpy array 
        Firing rate function evaluated on SEM grid

    """
    
    if not isinstance(center, np.ndarray):
        center, width, amp=np.array([center]), np.array([width]), np.array([amp])
    return np.maximum(np.sum(np.asarray([amp[i]*np.exp(-(x-center[i])**2/width[i]**2) for i in range(np.size(center))]),axis=0),0)
    
def sinus(x, bias=1000, amp=300, freq=np.pi):
    """ Rectified sinusoid
    

    Parameters
    ----------
    x : numpy array, dtype=float
        SEM grid points
    bias : float
        Bias. The default is 1000.
    amp : float
        Amplitude of the sinusoid. The default is 300.
    freq : float
        Frequency of the sinusoid. The default is np.pi.

    Returns
    -------
    numpy array 
        Firing rate function evaluated on SEM grid

    """
    return np.maximum(amp*np.sin(freq*x)+bias,0)       

def cos_square(x, amp=1000, freq=np.pi/2, bias=0):
    """Cosine-squareed peq model FR with given amplitude and frequency
    

    Parameters
    ----------
    x : numpy array, dtype=float
        SEM grid points
    amp : float
        Amplitude. The default is 1000.
    freq : float
        Frequency. The default is np.pi/2.
    bias : float
        Bias. The default is 0.

    Returns
    -------
    numpy array 
        Firing rate function evaluated on SEM grid

    """
    return np.maximum(amp*(np.cos(x*freq))**2+x*bias,0)
    