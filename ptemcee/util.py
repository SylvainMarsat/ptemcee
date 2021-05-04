#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import, unicode_literals

__all__ = ['_ladder', 'get_acf', 'get_integrated_act', 'thermodynamic_integration_log_evidence']

import numpy as np


def mod_interval(x, interval=None):
    """
    Remainder of x (can be a vector) on an interval ]a,b]
    Args:
      x          # Input data, can be a vector (no check)
    Kwargs:
      interval   # Target interval, list [a,b] for mod in ]a,b]
                   (default None, ignore)
    """
    if interval is None:
        return x
    else:
        a, b = interval
        return b - np.remainder(b-x, b-a)

def mod_diff_interval(x, y, interval=None):
    """
    Mod-diff x-y (float or 1-d array) for list of intervals ]a,b]
    Returns the difference with the shortest path taking into account mod
    Args:
      x          # Input data, can be a vector (no check)
      y          # Input data, can be a vector (no check)
    Kwargs:
      interval   # Target interval, list [a,b] for mod in ]a,b]
                   (default None, ignore)
    """
    if interval is None:
        return x - y
    else:
        a, b = interval
        modx = mod_interval(x, interval=interval)
        mody = mod_interval(y, interval=interval)
        mod_diff = mody - modx
        diffs = np.array([mod_diff - (b-a), mod_diff, mod_diff + (b-a)])
        n = len(x)
        return diffs[np.argmin(np.abs(diffs), axis=0), np.arange(n)]

def mod_arr(x, list_mod=None):
    """
    Remainder of x (2-d array) on a list of intervals ]a,b]
    Applies mod_interval with list_mod[i] on each column x[:,i]
    Args:
      x          # Input data, 2d array (no check)
    Kwargs:
      list_mod   # List of target intervals for each dimension: None to ignore,
                   or list [a,b] for mod in ]a,b]
                   (default None, ignore entirely)
    """
    if list_mod is None:
        return x
    else:
        x_mod = x.copy()
        ndim = x.shape[1]
        for i in range(ndim):
            x_mod[:,i] = mod_interval(x[:,i], interval=list_mod[i])
        return x_mod

def mod_diff_arr(x, y, list_mod=None):
    """
    Mod-diff x-y (2-d array) for list of intervals ]a,b]
    Returns the difference with the shortest path taking into account mod
    Applies mod_diff_interval with list_mod[i] on each column x[:,i], y[:,i]
    Args:
      x          # Input data, 2d array (no check)
      y          # Input data, 2d array (no check)
    Kwargs:
      list_mod   # List of target intervals for each dimension: None to ignore,
                   or list [a,b] for mod in ]a,b]
                   (default None, ignore entirely)
    """
    if list_mod is None:
        return x - y
    else:
        diff_mod = np.zeros_like(x)
        ndim = x.shape[1]
        for i in range(ndim):
            diff_mod[:,i] = mod_diff_interval(x[:,i], y[:,i], interval=list_mod[i])
        return diff_mod

def _ladder(betas):
    """
    Convert an arbitrary iterable of floats into a sorted numpy array.

    """

    betas = np.array(betas)
    betas[::-1].sort()
    return betas


def get_acf(x, axis=0, fast=False):
    """
    Estimate the autocorrelation function of a time series using the FFT.

    :param x:
        The time series. If multidimensional, set the time axis using the
        ``axis`` keyword argument and the function will be computed for every
        other axis.

    :param axis: (optional)
        The time axis of ``x``. Assumed to be the first axis if not specified.

    :param fast: (optional)
        If ``True``, only use the largest ``2^n`` entries for efficiency.
        (default: False)

    """
    x = np.atleast_1d(x)
    m = [slice(None), ] * len(x.shape)

    # For computational efficiency, crop the chain to the largest power of
    # two if requested.
    if fast:
        n = int(2 ** np.floor(np.log2(x.shape[axis])))
        m[axis] = slice(0, n)
        x = x
    else:
        n = x.shape[axis]

    # Compute the FFT and then (from that) the auto-correlation function.
    f = np.fft.fft(x - np.mean(x, axis=axis), n=2 * n, axis=axis)
    m[axis] = slice(0, n)
    acf = np.fft.ifft(f * np.conjugate(f), axis=axis)[tuple(m)].real
    m[axis] = 0
    return acf / acf[tuple(m)]


def get_integrated_act(x, axis=0, window=50, fast=False):
    """
    Estimate the integrated autocorrelation time of a time series.

    See `Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ on
    MCMC and sample estimators for autocorrelation times.

    :param x:
        The time series. If multidimensional, set the time axis using the
        ``axis`` keyword argument and the function will be computed for every
        other axis.

    :param axis: (optional)
        The time axis of ``x``. Assumed to be the first axis if not specified.

    :param window: (optional)
        The size of the window to use. (default: 50)

    :param fast: (optional)
        If ``True``, only use the largest ``2^n`` entries for efficiency.
        (default: False)

    """
    # Compute the autocorrelation function.
    f = get_acf(x, axis=axis, fast=fast)

    # Special case 1D for simplicity.
    if len(f.shape) == 1:
        return 1 + 2 * np.sum(f[1:window])

    # N-dimensional case.
    m = [slice(None), ] * len(f.shape)
    m[axis] = slice(1, window)
    tau = 1 + 2 * np.sum(f[tuple(m)], axis=axis)

    return tau


def thermodynamic_integration_log_evidence(betas, logls):
    """
    Thermodynamic integration estimate of the evidence.

    :param betas: The inverse temperatures to use for the quadrature.

    :param logls:  The mean log-likelihoods corresponding to ``betas`` to use for
        computing the thermodynamic evidence.

    :return ``(logZ, dlogZ)``: Returns an estimate of the
        log-evidence and the error associated with the finite
        number of temperatures at which the posterior has been
        sampled.

    The evidence is the integral of the un-normalized posterior
    over all of parameter space:

    .. math::

        Z \\equiv \\int d\\theta \\, l(\\theta) p(\\theta)

    Thermodymanic integration is a technique for estimating the
    evidence integral using information from the chains at various
    temperatures.  Let

    .. math::

        Z(\\beta) = \\int d\\theta \\, l^\\beta(\\theta) p(\\theta)

    Then

    .. math::

        \\frac{d \\log Z}{d \\beta}
        = \\frac{1}{Z(\\beta)} \\int d\\theta l^\\beta p \\log l
        = \\left \\langle \\log l \\right \\rangle_\\beta

    so

    .. math::

        \\log Z(1) - \\log Z(0)
        = \\int_0^1 d\\beta \\left \\langle \\log l \\right\\rangle_\\beta

    By computing the average of the log-likelihood at the
    difference temperatures, the sampler can approximate the above
    integral.
    """
    if len(betas) != len(logls):
        raise ValueError('Need the same number of log(L) values as temperatures.')

    order = np.argsort(betas)[::-1]
    betas = betas[order]
    logls = logls[order]

    betas0 = np.copy(betas)
    if betas[-1] != 0:
        betas = np.concatenate((betas0, [0]))
        betas2 = np.concatenate((betas0[::2], [0]))

        # Duplicate mean log-likelihood of hottest chain as a best guess for beta = 0.
        logls2 = np.concatenate((logls[::2], [logls[-1]]))
        logls = np.concatenate((logls, [logls[-1]]))
    else:
        betas2 = np.concatenate((betas0[:-1:2], [0]))
        logls2 = np.concatenate((logls[:-1:2], [logls[-1]]))

    logZ = -np.trapz(logls, betas)
    logZ2 = -np.trapz(logls2, betas2)
    return logZ, np.abs(logZ - logZ2)
