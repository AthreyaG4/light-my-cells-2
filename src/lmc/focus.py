"""Best-focus slice of a z-stack. Shared by data prep (reconcile) and, later, inference.

The two focus measures are the ones provided with the challenge (unchanged). The best slice is
the one with the highest score: normalized variance for transmitted-light channels, auto
correlation for fluorescence (organelle) channels.
"""

import numpy as np

from lmc.config import INPUT_MODALITIES


def zfocus_normalized_variance(arr):
    """
    Get the best Zfocus of multipleZ images using normalized variance (recommended for transmitted light images
    Args :
        - An image array of shape TCZYX with T,C= 1
    Out:
        - An image array of best Zfocus with shape ZYX with Z=1
    """
    arr = arr[0,0,...]
    *_, y, x = arr.shape
    mean = arr.mean(axis=(-2, -1))[:, None, None]

    z_plane = ((arr - mean)**2).sum(axis=(-2, -1))[:, None, None] / (y * x * mean)
    return z_plane


def zfocus_auto_correlation(arr):
    """
    Get the best Zfocus of multiple Z stack images using auto correlation, recommended for fluorescence images.
    Args :
        - An image array of shape TCZYX with T,C= 1
    Out:
        - An image array of best Zfocus with shape ZYX with Z=1
    """
    arr = (arr[0,0,...] - arr[0,0,...].mean()) / arr.std()
    auto_corr = (((arr[:, :-1, :] * arr[:, 1:, :]).sum(axis=(-2, -1)))
                 - (arr[:, :-2, :] * arr[:, 2:, :]).sum(axis=(-2, -1)))
    return auto_corr


def best_slice(stack, channel):
    """Position (0-based) of the sharpest slice in a z-stack, and the method used.

    stack: array of shape (Z, Y, X) with the slices in z order; channel: e.g. "DIC" or "Nucleus".
    """
    stack = np.asarray(stack, dtype=np.float64)[None, None]  # TCZYX, T=C=1
    if channel in INPUT_MODALITIES:
        method, scores = "normalized_variance", zfocus_normalized_variance(stack)
    else:
        method, scores = "auto_correlation", zfocus_auto_correlation(stack)
    scores = scores.ravel()
    if np.isnan(scores).all():
        raise ValueError("every slice scored NaN")
    return int(np.nanargmax(scores)), method
