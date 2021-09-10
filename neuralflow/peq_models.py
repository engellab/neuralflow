#  
#

"""This is a part of neuralflow package/EnergyModel class.
This source file contains template peq functions.
Equilibirum probability distribution is related to potential U(x)=-log(peq).
"""


import numpy as np
from neuralflow.utilities.spline_padding import add_anchor_point
from scipy.interpolate import CubicSpline, PPoly

def custom(x,w,lambdafunc=None):
    """Custom model (for example, for real data fitting).
    
    Parameters
    ----------
    x : numpy array (N,), dtype=float
        Grid points in which the model will be evaluated. N is the number of grid points.
    w : numpy array (N,), dtype=float
        Weights used to evaluate integrals by the Gaussian quadrature.
    lambdafunc: function or None
        Either supply a function, or leave None if the model of the generated data is not known, in which case 
        x and w are ignored.
    
    Returns
    -------
    peq : numpy array (N,), dtype=float
        Probability density distribution evaluated at grid ponits x.
    """
    if lambdafunc is not None:
        peq=lambdafunc(x)
        peq /= sum(w*peq)
    else:
        peq=None
    return peq

def cos_square(x, w, xbegin=-1.0, xend=1.0):
    """Cosine-squareed model.

        y, L --> peq(y, L) ~ cos( y*pi/L )**2
        
        integral peq(x) dx =1
        y - x centered on the middle of the domain
        L - domain length

    Parameters
    ----------
    x : numpy array (N,), dtype=float
        Grid points in which the model will be evaluated. N is the number of grid points.
    w : numpy array (N,), dtype=float
        Weights used to evaluate integrals by the Gaussian quadrature.    
    xbegin: float
        Left boundary of the x domain. The default is -1.
    xend: float
        Right boundary of the x-domain. The default is 1.

    Returns
    -------
    peq : numpy array (N,), dtype=float
        Probability density distribution evaluated at grid ponits x.
    """
    y = x-(xbegin+xend)/2
    L = xend-xbegin
    peq = (np.cos(y*np.pi/L))**2
    # normalization
    peq /= sum(w*peq)
    return peq

def cos_fourth_power(x, w, xbegin=-1.0, xend=1.0):
    """Cosine^4 peq model.

    y, L --> peq(y, L) ~ cos( y*pi/L )**2, integral peq(x) dx =1
        where y - x centered on the middle of the domain, L - domain length

    Parameters
    ----------
    x : numpy array (N,), dtype=float
        Grid points in which the model will be evaluated. N is the number of grid points.
    w : numpy array (N,), dtype=float
        Weights used to evaluate integrals by the Gaussian quadrature. 
    xbegin: float
        Left boundary of the x domain. The default is -1.
    xend: float
        Right boundary of the x-domain. The default is 1.

    Returns
    -------
    peq : numpy array, dtype=float
        Probability density distribution evaluated at grid ponits x.
    """
    y = x-(xbegin+xend)/2
    L = xend-xbegin
    peq = (np.cos(y*np.pi/L))**4
    # normalization
    peq /= sum(w*peq)
    
    return peq

def jump_spline2(x, w,
               interp_x=[-1, -0.5, 0, 0.3, 1], 
               interp_y=[0, 4, 0, 1, -1],
               bc_left = [(1, 0), (1, 0)],
               bc_right = [(1, 0), (1, 0)]):
    """Stepping model.
     
    Parameters
    ----------
    x : numpy array (N,), dtype=float
        Grid points in which the model will be evaluated. N is the number of grid points.
    w : numpy array (N,), dtype=float
        Weights used to evaluate integrals by the Gaussian quadrature. 
    interp_x: numpy array (5,), dtype=float, or list
        The x positions of the boundaries, maixma and minima of the potential,
        sorted from the left to the right. Contains 5 values: the left boundary,
        the first maximum, the minimum, the second maximum and the right boundary.
        The default is [-1, -0.5, 0, 0.3, 1], which corresponds to the stepping
        potential used in M. Genkin, O. Hughes, T.A. Engel paper.
    interp_y: numpy array (5,), dtype=float, or list
        The corresponding y-values of the potential at the points specified by
        interp_x. The default is [0, 4, 0, 1, -1], which corresponds to the stepping
        potential used in M. Genkin, O. Hughes, T.A. Engel paper.
    bc_left: list
        A list that contains two tuples that specify boundary conditions for the potential
        on the left boundary. The format is the same as in bc_type argument of 
        scipy.interpolate.CubicSpline function. The default is [(1, 0), (1, 0)],
        which corresponds to a zero-derivative (Neumann) BC for the potential.
    bc_right: list
        Same as bc_left, but for a right boundary. The default is [(1, 0), (1, 0)],
        which corresponds to a zero-derivative (Neumann) BC for the potential.

    Returns
    -------
    peq : numpy array, dtype=float
        Probability density distribution evaluated at grid ponits x.
    """
    xv = np.array(interp_x)
    yv = np.array(interp_y)
    cs = np.zeros((xv.shape[0]+1, 4),dtype=x.dtype);
    
    #Find additonal anchoring points on left and right boundaries
    x_add_l, y_add_l = add_anchor_point(xv[:3],yv[:3])
    x_add_r, y_add_r = add_anchor_point(xv[-3:][::-1],yv[-3:][::-1])
    
    #Add these poionts to the x and y arrays
    xv_new = np.concatenate(([xv[0], x_add_l],xv[1:-1],[x_add_r, xv[-1]])) 
    yv_new = np.concatenate(([yv[0], y_add_l],yv[1:-1],[y_add_r, yv[-1]]))
    
    #Use three points for boundary splines, and two points for splines in the bulk
    cs[0:2,:] = CubicSpline(xv_new[:3], yv_new[:3], bc_type=bc_left).c.T
    for i in range(1,xv.shape[0]-2):
        cs[i+1,:] = CubicSpline(xv[i:i+2], yv[i:i+2], bc_type=[(1, 0), (1, 0)]).c.T
    cs[-2:] = CubicSpline(xv_new[-3:], yv_new[-3:], bc_type=bc_right).c.T
    
    poly = PPoly(cs.T, xv_new)
    peq = np.exp(-poly(x))
    # normalization
    peq /= sum(w*peq)
    return peq

def linear_pot(x,w,slope=1):
    """Ramping model derived from a linear potential
    
    V(x)=slope*x
    peq(x) = exp (-slope*x) / || exp (-slope*x) ||
        
    Parameters
    ----------
    x : numpy array (N,), dtype=float
        Grid points in which the model will be evaluated. N is the number of grid points.
    w : numpy array (N,), dtype=float
        Weights used to evaluate integrals by the Gaussian quadrature. 
    slope: float
        Slope of the potential function.
     
    Returns
    -------
    peq : numpy array, dtype=float
        Probability density distribution evaluated at grid ponits x.
    """
    V=slope*x
    peq = np.exp(-V)  
    # normalization
    peq /= sum(w*peq)
    
    return peq

def single_well(x, w, xmin=0, miu=10.0, sig=0):
    """
    Peq derived from a single well quadratic potential.

        peq(x) = exp[ -U(x) ]
        
        y = x - xmin
        
        U(y) = miu * y**2 + sig*y
        
        integral peq(x) dx =1

    Parameters
    ----------
    x : numpy array (N,), dtype=float
        Grid points in which the model will be evaluated. N is the number of grid points.
    w : numpy array (N,), dtype=float
        Weights used to evaluate integrals by the Gaussian quadrature.     
    xmin : float
        position of the minimum for sig=0 case. The default is 0.
    miu : float
        curvature (steepness) of the potential. The default is 10.   
    sig : float
        assymetry parameter, to bias left vs. rights sides of the potential. The default is 0.

    Returns
    -------
    peq : numpy array, dtype=float
        Probability density distribution evaluated at grid ponits x.
    """
    
    y = x - xmin

    peq = np.exp(-( 0.5*miu*y**2 + sig*y ))
    # normalization
    peq /= sum(w*peq)
    
    return peq

def double_well(x, w, xmax=0, xmin=0.3, depth=4.0, sig=0):
    """
    Peq derived from a double-well potential.

        peq(x) = exp[ -U(x) ]
        
        y = x - xmax
        ymin = xmin - xmax
        
        U(y) = depth * (y/ymin)**4 - 2*depth * (y/ymin)**2 +sig*y
        
        integral peq(x) dx =1

    Parameters
    ----------
    x : numpy array (N,), dtype=float
        Grid points in which the model will be evaluated. N is the number of grid points.
    w : numpy array (N,), dtype=float
        Weights used to evaluate integrals by the Gaussian quadrature. 
    xmax : float
        position of the maximum of the potential in the symmetrix case (sig=0) 
    xmin : float
        position of the two symmetric minima for sig=0 case. The default is 0. 
    depth : float
        depth of the potential wells (U_max - U_min) for the symmetric case (sig=0). The default is 4. 
    sig : float
        assymetry parameter, to bias one well over another. The default is 0.

    Returns
    -------
    peq : numpy array (N,), dtype=float
        Probability density distribution evaluated at grid ponits x.
    """
    
    y = x - xmax
    ymin = xmin - xmax

    peq = np.exp(-( depth*(y/ymin)**4 - 2*depth*(y/ymin)**2 + sig*y ))
    # normalization
    peq /= sum(w*peq)
    
    return peq

def uniform(x, w):
    """
    Uniform peq model, derived from a constant potential.

        peq = const
        
        integral peq(x) dx =1

    Parameters
    ----------
    x : numpy array (N,), dtype=float
        Grid points in which the model will be evaluated. N is the number of grid points.
    w : numpy array (N,), dtype=float
        Weights used to evaluate integrals by the Gaussian quadrature. 
    Returns
    -------
    peq : numpy array (N,), dtype=float
        Probability density distribution evaluated at grid ponits x.
    """

    peq = np.ones(x.shape)
    # normalization
    peq /= sum(w*peq)
    
    return peq



peq_model_mixtures = {
    'cos_square': cos_square,
    'single_well': single_well,
    'double_well': double_well,
    'uniform': uniform
}


def mixture(x, w, theta=0.5, model1={'model': 'cos_square', 'params': {}}, model2={'model': 'double_well', 'params': {}}):
    """
    Convex mixture of two models.

        peq = theta*peq_model_1 + (1-theta)*peq*model_2
        
        integral peq(x) dx =1

    Parameters
    ----------
    x : numpy array (N,), dtype=float
        Grid points in which the model will be evaluated.
    w : numpy array (N,), dtype=float
        Weights used to evaluate integrals by the Gaussian quadrature.    
    theta : float
        convex weight, 0<=theta<=1. The default is 0.5.
    model1 : dictionary.
        Dictionary that contains model name (one of the built-in models) and the corresponding parameters. 
        The default is {'model': 'cos_square', 'params': {}}
    model2 : dictionary.
        Dictionary that contains model name (one of the built-in models) and the corresponding parameters. 
        The default is {'model': 'double_well', 'params': {}}

    Returns
    -------
    peq : numpy array (N,), dtype=float
        Probability density distribution evaluated at grid ponits x.
    """
    assert 0 <= theta <= 1
    
    if not model1['model'] in peq_model_mixtures:
        raise ValueError("peq_model type for model mixtures should be one of %s, "
                         "%s was given."
                         % (peq_model_mixtures.keys(),  model1['model']))
        
    if not model2['model'] in peq_model_mixtures:
        raise ValueError("peq_model type for model mixtures should be one of %s, "
                         "%s was given."
                         % (peq_model_mixtures.keys(),  model2['model']))
    
    peq1 = peq_model_mixtures[model1['model']](x, w, **model1['params'])
    peq2 = peq_model_mixtures[model2['model']](x, w, **model2['params'])
    
    peq  = theta*peq1 + (1-theta)*peq2
    
    # normalization
    peq /= sum(w*peq)
    
    
    return peq

