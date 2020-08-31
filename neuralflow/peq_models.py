#  
#

"""
The built-in models for the peq probability density
submodule for the energy_model module.
"""


import numpy as np

def custom(x,w,lambdafunc=None):
    """
    Custom model (for example, for real data fitting).
    Either supply a function, or leave None if the model of the generated data is not known
    """
    if lambdafunc is not None:
        peq=lambdafunc(x)
        peq /= sum(w*peq)
    else:
        peq=None
    return peq

def cos_square(x, w, xbegin=-1.0, xend=1.0):
    """
    Cosine-squareed peq model.

        y, L --> peq(y, L) ~ cos( y*pi/L )**2
        
        integral peq(x) dx =1
        y - x centered on the middle of the domain
        L - domain length

    Parameters
    ----------
    x : array
        An array with shape N (number of x-nodes) giving the
        grid-points to evaluate the probability density function.

    w : double
        Weights used to evaluate integrals by Gauss quadrature.
        
    xbegin, xend : double, double
        Boundaries of the x-domain.

    Returns
    -------
    peq : array
        Equilibrium probability-density function evaluated at grid-ponits x.
    """
    y = x-(xbegin+xend)/2
    L = xend-xbegin
    peq = (np.cos(y*np.pi/L))**2
    # normalization
    peq /= sum(w*peq)
    
    return peq

def cos_fourth_power(x, w, xbegin=-1.0, xend=1.0):
    """
    Cosine^4 peq model.

        y, L --> peq(y, L) ~ cos( y*pi/L )**2
        
        integral peq(x) dx =1
        y - x centered on the middle of the domain
        L - domain length

    Parameters
    ----------
    x : array
        An array with shape N (number of x-nodes) giving the
        grid-points to evaluate the probability density function.

    w : double
        Weights used to evaluate integrals by Gauss quadrature.
        
    xbegin, xend : double, double
        Boundaries of the x-domain.

    Returns
    -------
    peq : array
        Equilibrium probability-density function evaluated at grid-ponits x.
    """
    y = x-(xbegin+xend)/2
    L = xend-xbegin
    peq = (np.cos(y*np.pi/L))**4
    # normalization
    peq /= sum(w*peq)
    
    return peq

def single_well(x, w, xmin=0, miu=10.0, sig=0):
    """
    Single-well quadratice potential model for peq.

        peq(x) = exp[ -U(x) ]
        
        y = x - xmin
        
        U(y) = miu * y**2 + sig*y
        
        integral peq(x) dx =1

    Parameters
    ----------
    x : array
        An array with shape N (number of x-nodes) giving the
        grid-points to evaluate the probability density function.

    w : double
        Weights used to evaluate integrals by Gauss quadrature.
        
    xmin : double
        position of the minimum for sig=0 case
        
    miu : double
        curvature of the potential
        
    sig : double
        assymetry parameter, to bias left vs. rights sides of the potential

    Returns
    -------
    peq : array
        Equilibrium probability-density function evaluated at grid-ponits x.
    """
    
    y = x - xmin

    peq = np.exp(-( 0.5*miu*y**2 + sig*y ))
    # normalization
    peq /= sum(w*peq)
    
    return peq

def double_well(x, w, xmax=0, xmin=0.3, depth=4.0, sig=0):
    """
    Double-well potential model for peq.

        peq(x) = exp[ -U(x) ]
        
        y = x - xmax
        ymin = xmin - xmax
        
        U(y) = depth * (y/ymin)**4 - 2*depth * (y/ymin)**2 +sig*y
        
        integral peq(x) dx =1

    Parameters
    ----------
    x : array
        An array with shape N (number of x-nodes) giving the
        grid-points to evaluate the probability density function.

    w : double
        Weights used to evaluate integrals by Gauss quadrature.
        
    xmax : double
        position of the maximum of the potential in the symmetrix case (sig=0)
        
    xmin : double
        position of the two symmetric minima for sig=0 case
        
    depth : double
        depth of the potential wells (U_max - U_min) for the symmetric case (sig=0)
        
    sig : double
        assymetry parameter, to bias one well over another

    Returns
    -------
    peq : array
        Equilibrium probability-density function evaluated at grid-ponits x.
    """
    
    y = x - xmax
    ymin = xmin - xmax

    peq = np.exp(-( depth*(y/ymin)**4 - 2*depth*(y/ymin)**2 + sig*y ))
    # normalization
    peq /= sum(w*peq)
    
    return peq

def uniform(x, w):
    """
    Uniform peq model.

        peq = const
        
        integral peq(x) dx =1

    Parameters
    ----------
    x : array
        An array with shape N (number of x-nodes) giving the
        grid-points to evaluate the probability density function.

    w : double
        Weights used to evaluate integrals by Gauss quadrature.

    Returns
    -------
    peq : array
        Equilibrium probability-density function evaluated at grid-ponits x.
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
    x : array
        An array with shape N (number of x-nodes) giving the
        grid-points to evaluate the probability density function.

    w : double
        Weights used to evaluate integrals by Gauss quadrature.
        
    theta : double
        convex weighting parameter 0<=theta<=1
    
    model_1, model_2 : dictionary
        'model': model type, should be in the set of defined models
        'param': model parameters for the selected model type

    Returns
    -------
    peq : array
        Equilibrium probability-density function evaluated at grid-ponits x.
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

