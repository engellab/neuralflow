# -*- coding: utf-8 -*-
"""Template firing rate functions."""

import numpy as np

# Minimum firing rate in Hz to prevent log(f(x)) attaining too low values
min_firing_rate = 1


def custom(x, lambdafunc=None):
    """Custom fr model.
    Either supply a function, or leave None if the model of the generated data
    is not known (and define fr manually after).


    Parameters
    ----------
    x : numpy array, dtype=float
        SEM grid points
    lambdafunc : function object.
        The default is None.

    Returns
    -------
    fr : numpy array
        Firing rate function evaluated on SEM grid

    """
    if lambdafunc is not None:
        fr = lambdafunc(x)
        return np.maximum(fr, min_firing_rate)
    else:
        return None


def rectified_linear(x, slope=50.0, x_thresh=-1.0):
    """Rectified-linear firing rate model.
    r(x, slope, thresh) = max[slope*(x - x_thresh), 0]

    Parameters
    ----------
    x : numpy array, dtype=float
        SEM grid points
    slope : float
        Firing-rate slope parameter. The default is 50.0.
    threshold : float
        Firing threshold parameter. The default is -1.0.

    Returns
    -------
    numpy array
        Firing rate function evaluated on SEM grid
    """

    return np.maximum(slope * (x - x_thresh), min_firing_rate)


def linear(x, slope=50.0, bias=2):
    """Linear firing rate model.
    r(x, slope, bias) = max[slope * x + bias, 0]

    Parameters
    ----------
    x : numpy array, dtype=float
        SEM grid points
    slope : float
        Firing-rate slope parameter. The default is 50.0.
    bias : float
        Firing threshold parameter. The default is 2.

    Returns
    -------
    numpy array
        Firing rate function evaluated on SEM grid

    """
    return np.maximum(slope * x + bias, min_firing_rate)


def peaks(x, center=np.array([-0.5, 0.5]), width=np.array([0.2, 0.3]),
          amp=np.array([100, 80])):
    """Sum of gaussian peaks
    f= SUM(A*exp((x-x0)^2/w^2))

    Parameters
    ----------
    x : numpy array, dtype=float
        SEM grid points
    center : numpy array, dtype=float
        Centers of Gaussian peaks. The default is np.array([-0.5,0.5]).
    width : numpy array, dtype=float
        Widths of Gaussian peaks. The default is np.array([0.2,0.3]).
    amp : numpy array, dtype=float
        Magnitudes of Gaussian peaks. The default is np.array([100,80]).

    Returns
    -------
    numpy array
        Firing rate function evaluated on SEM grid

    """

    if not isinstance(center, np.ndarray):
        center, width, amp = np.array(center), np.array(width), np.array(amp)
    out = np.asarray([
        amp[i]*np.exp(-(x-center[i])**2/width[i]**2)
        for i in range(np.size(center))
    ])
    return np.maximum(np.sum(out, axis=0), min_firing_rate)


def sinus(x, bias=100, amp=30, freq=np.pi):
    """ Rectified sinusoid


    Parameters
    ----------
    x : numpy array, dtype=float
        SEM grid points
    bias : float
        Bias. The default is 100.
    amp : float
        Amplitude of the sinusoid. The default is 30.
    freq : float
        Frequency of the sinusoid. The default is np.pi.

    Returns
    -------
    numpy array
        Firing rate function evaluated on SEM grid

    """
    return np.maximum(amp*np.sin(freq*x)+bias, min_firing_rate)


def cos_square(x, amp=1000, freq=np.pi/2, bias=0):
    """Cosine-squareed model with a given amplitude and frequency


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
    return np.maximum(amp*(np.cos(x*freq))**2+x*bias, min_firing_rate)


firing_model_types_ = {
    'custom': custom,
    'rectified_linear': rectified_linear,
    'linear': linear,
    'peaks': peaks,
    'sinus': sinus,
    'cos_square': cos_square
}
