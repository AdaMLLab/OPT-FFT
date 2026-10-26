"""Moments of continuous CO96/R70 priors restricted to spatial intervals.

The exact integration algorithm takes these moments as input. This module
computes numerical approximations to them; moment error is separate from
FFT roundoff. It never defines a fixed atomic prior for portfolio evaluation.

For degree D and K regions, the angular quadrature and nonuniform Fourier
summation use O(K*(D+log(1/tol))*log(D/tol)*log(1/tol)) operations with fixed
oversampling. The tolerance controls Fourier truncation and the central
tail approximation.
"""

from functools import lru_cache

import numpy as np
from scipy.fft import ifft
from scipy.special import roots_legendre


@lru_cache(maxsize=16)
def _legendre_rule(order):
    return roots_legendre(order)


def _positive_fourier_sums(phase, weight, degree, tol):
    """Positive nonuniform Fourier sums, using oversampled FFTs."""
    size = 1 << max(2, (4 * degree - 1).bit_length())
    step = 2 * np.pi / size
    nearest = np.rint(phase / step).astype(np.intp)
    offset = (phase - step * nearest) * (size / np.pi)
    terms, remainder = 0, np.exp(np.pi / 4)
    while remainder > tol:
        terms += 1
        remainder *= (np.pi / 4) / terms
    power = weight.copy()
    factor = np.ones(degree + 1, dtype=complex)
    increment = 1j * np.arange(degree + 1) * (np.pi / size)
    sums = np.zeros(degree + 1, dtype=complex)
    for order in range(terms):
        grid = np.bincount(nearest, weights=power, minlength=size)
        sums += factor * (ifft(grid)[:degree + 1] * size)
        power *= offset
        factor *= increment / (order + 1)
    return sums


@lru_cache(maxsize=2)
def _angular_rule(degree, order):
    panels = max(1, int(np.ceil(np.pi * degree / 4)))
    edges = np.linspace(0, np.pi, panels + 1)
    z, w = _legendre_rule(order)
    half = np.diff(edges)[:, None] / 2
    theta = (edges[:-1, None] + (z + 1) * half).ravel()
    weights = (half * w).ravel()
    return theta, weights


def prior_mass(prior, lower, upper):
    """Continuous prior mass of a region; no quadrature is used here."""
    if prior == "co96":
        if not 0 <= lower < upper <= 1:
            raise ValueError("CO96 region must lie in [0, 1]")
        if lower + upper > 1:
            lower, upper = 1 - upper, 1 - lower
        return float(2 / np.pi * (np.arcsin(np.sqrt(upper)) - np.arcsin(np.sqrt(lower))))
    if prior != "r70" or not -1 <= lower < upper <= 1:
        raise ValueError("R70 region must lie in [-1, 1]")
    a = 1 + np.log(6.6)
    if lower == -upper:
        return float(np.log(a) / np.log(a - np.log(upper)))
    if upper <= 0:
        lower, upper = -upper, -lower
    if lower <= 0:
        raise ValueError("an R70 region containing zero must be symmetric")
    low_log, high_log = a - np.log(upper), a - np.log(lower)
    return float(np.log(a) / 2 * np.log1p(np.log(upper / lower) / low_log)
                 / (np.log(low_log) * np.log(high_log)))


def _central_r70(radius, degree, tol, oversample):
    """Conditional moments on [-radius, radius], resolving the singularity.

    beta=radius*sin(phi) reduces even moments to cosine integrals. The
    unresolved part at phi=0 is bounded at the moment level and shrinks
    with degree and tolerance; it is not a fixed tail atom in the test.
    """
    a = 1 + np.log(6.6) - np.log(radius)
    cutoff = np.sqrt(tol) / (2 * max(2, degree))
    edges = [cutoff]
    while edges[-1] < np.pi / 2:
        edges.append(min(2 * edges[-1], edges[-1] + 4 / max(2, degree), np.pi / 2))
    edges = np.asarray(edges)
    order = int(np.ceil(oversample * (np.log(8192) + np.log(max(2, degree))
                                     - 1.5 * np.log(tol)) / np.log(4)))
    z, w = _legendre_rule(order)
    half = np.diff(edges)[:, None] / 2
    phi = (edges[:-1, None] + (z + 1) * half).ravel()
    denominator = a - np.log(np.sin(phi))
    weights = ((half * w).ravel() * np.log(a) * np.cos(phi)
               / (np.sin(phi) * denominator * np.log(denominator)**2))
    tail = np.log(a) / np.log(a - np.log(np.sin(cutoff)))
    moments = np.zeros(degree + 1)
    moments[::2] = tail + _positive_fourier_sums(2 * phi, weights, degree // 2, tol / 16).real
    moments[2::4] *= -1
    moments[0] = 1
    return moments


def panel_moments(prior, lower, upper, degree, *, tol=1e-13, oversample=1.0):
    """Return conditional Chebyshev moments through degree, including m_0=1.

    m_k = E_F[T_k((2*coordinate-lower-upper)/(upper-lower)) | region].
    A caller may replace this function with exact/precomputed prior moments.
    Numerical integration here approximates these continuous-prior constants.
    """
    if not np.isfinite(degree) or degree < 0 or int(degree) != degree:
        raise ValueError("degree must be a nonnegative integer")
    if not np.isfinite(tol) or not 0 < tol < .5:
        raise ValueError("moment tolerance must be strictly between 0 and 0.5")
    if not np.isfinite(oversample) or oversample < 1:
        raise ValueError("moment oversample must be finite and at least 1")
    prior_mass(prior, lower, upper)  # Validate the region.
    degree = int(degree)
    if degree == 0:
        return np.ones(1)
    if prior == "co96" and lower + upper > 1:
        result = panel_moments(prior, 1-upper, 1-lower, degree, tol=tol, oversample=oversample)
        result[1::2] *= -1
        return result
    if prior == "r70" and upper <= 0:
        result = panel_moments(prior, -upper, -lower, degree, tol=tol, oversample=oversample)
        result[1::2] *= -1
        return result
    if prior == "r70" and lower == -upper:
        return _central_r70(upper, degree, tol, oversample)
    if prior == "co96" and lower == 0 and upper == 1:
        result = np.zeros(degree + 1)
        result[0] = 1
        return result
    order = int(np.ceil(oversample * max(16, np.log(1/tol) / np.log(8))))
    theta, quadrature = _angular_rule(degree, order)
    half, middle = (upper-lower)/2, (upper+lower)/2
    coordinate = middle + half*np.cos(theta)
    if prior == "co96":
        if lower == 0:
            # Cancel the endpoint square root analytically before evaluation.
            density = np.sqrt(upper)*np.sin(theta/2)/(np.pi*np.sqrt(1-coordinate))
        else:
            density = half*np.sin(theta)/(np.pi*np.sqrt(coordinate*(1-coordinate)))
    else:
        a = 1 + np.log(6.6)
        denominator = a - np.log(coordinate)
        density = (half*np.sin(theta)*np.log(a)
                   / (2*coordinate*denominator*np.log(denominator)**2))
    weights = quadrature*density
    weights /= weights.sum()
    result = _positive_fourier_sums(theta, weights, degree, tol/16).real
    result[0] = 1
    return result
