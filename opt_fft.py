"""OPT-FFT: Optimal Portfolio Testing with Fast Fourier Transforms.

Test a specified bounded mean using the 50/50 CO96/R70 hedge, as in Remark 2 in Orabona and Jun (2024).

CO96 denotes the Beta(1/2, 1/2) portfolio prior analyzed by Cover and Ordentlich
(1996). R70 denotes the Robbins (1970) prior adapted by Orabona and Jun (2024)
to betting fractions in [-1, 1].

The one-dimensional Prefix algorithm uses fast Fourier transforms (FFTs)
to compute cosine transforms.
See the paper for the priors, numerical qualifications, and complexity:

@article{orabon2026optimal,
  title={Optimal Priors for Betting Tests with Bounded Payoffs: An Exact Expected-Time Characterization},
  author={Orabona, Francesco},
  journal={arXiv preprint arXiv:2610.25821},
  year={2026}
}

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct, ifft
from scipy.special import roots_legendre


def _data(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("x must be a finite one-dimensional array in [0, 1]")
    return x


@lru_cache(maxsize=256)
def _nodes(n):
    return np.cos(np.pi * (np.arange(n) + 0.5) / n)


def _resample(values, n):
    """Chebyshev interpolation (or coefficient truncation), last axis."""
    old = values.shape[-1]
    if old == n:
        return values.copy()
    coef = dct(values, type=2, norm="ortho", axis=-1)
    return idct(coef[..., :n], type=2, n=n, norm="ortho", axis=-1) * np.sqrt(n / old)


def _restrict(weights, n):
    # Transpose of interpolation FROM the smaller grid, not interpolation of
    # a density.  The one-dimensional normalization factor is old / n.
    return _resample(weights, n) * (weights.shape[-1] / n)


def _unit_scale(values):
    scale = np.max(np.abs(values))
    if not np.isfinite(scale) or scale == 0:
        raise FloatingPointError("degenerate polynomial/functional in FFT recursion")
    return values / scale


def _moment_parameters(degree, tol, oversample):
    if not np.isfinite(degree) or int(degree) != degree or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    if not np.isfinite(tol) or not 0 < tol < 0.5:
        raise ValueError("moment tolerance must be strictly between 0 and 0.5")
    if not np.isfinite(oversample) or oversample < 1:
        raise ValueError("moment oversample must be finite and at least 1")
    return int(degree)


@lru_cache(maxsize=32)
def _legendre_rule(order):
    return roots_legendre(order)


def _r70_angular_quadrature(degree, tol, oversample):
    # beta=sin(phi): even Chebyshev moments become cosine Fourier modes.
    degree = max(2, degree)
    cutoff = np.sqrt(tol) / (2 * degree)
    edges = [cutoff]
    while edges[-1] < np.pi / 2:
        edges.append(min(2 * edges[-1], edges[-1] + 4 / degree, np.pi / 2))
    edges = np.asarray(edges)
    # The appendix bounds the total Gauss error by
    # 1024 * degree / sqrt(tol) * 4**(-order) <= tol/8.
    order = int(np.ceil(oversample * (np.log(8192) + np.log(degree)
                                    - 1.5 * np.log(tol)) / np.log(4)))
    z, w = _legendre_rule(order)
    half_width = np.diff(edges)[:, None] / 2
    phi = (edges[:-1, None] + (z + 1) * half_width).ravel()
    log_ratio = 1 + np.log(6.6) - np.log(np.sin(phi))
    normalizer = np.log(1 + np.log(6.6))
    weight = ((half_width * w).ravel() * normalizer * np.cos(phi)
              / (np.sin(phi) * log_ratio * np.log(log_ratio) ** 2))
    tail = normalizer / np.log(1 + np.log(6.6) - np.log(np.sin(cutoff)))
    return phi, weight, tail


def r70_quadrature(degree, *, tol=1e-13, oversample=1.0):
    """Positive quadrature for the continuous symmetric R70 law.

    Return positive beta nodes, total (both signs) weights, and the exact
    mass below the cutoff, represented at beta=0. The cutoff and panel
    order depend on degree and absolute moment tolerance. Weights sum to
    1-tail up to quadrature/roundoff error. This materialized rule is also
    useful for direct validation; r70_moments evaluates its modes by FFT.
    """
    degree = _moment_parameters(degree, tol, oversample)
    phi, weight, tail = _r70_angular_quadrature(degree, tol, oversample)
    return np.sin(phi), weight, tail


def _positive_fourier_sums(phase, weight, max_mode, tol):
    """Sum weight*exp(1j*k*phase), 0<=k<=max_mode, by oversampled FFTs.

    Phases lie in [0, pi] and weights are positive. Nearest-grid offsets
    are scaled to [-1, 1], and k times an unscaled offset is <= pi/4.
    The Taylor remainder is bounded by tol * sum(abs(weight)).
    """
    size = 1 << max(2, (4 * max_mode - 1).bit_length())
    step = 2 * np.pi / size
    nearest = np.rint(phase / step).astype(np.intp)
    offset = (phase - step * nearest) * (size / np.pi)
    # e**(pi/4)*(pi/4)**terms/terms! bounds each exponential error.
    terms, remainder = 0, np.exp(np.pi / 4)
    while remainder > tol:
        terms += 1
        remainder *= (np.pi / 4) / terms
    weighted_power = weight.copy()
    factor = np.ones(max_mode + 1, dtype=complex)
    increment = 1j * np.arange(max_mode + 1) * (np.pi / size)
    sums = np.zeros(max_mode + 1, dtype=complex)
    for r in range(terms):
        grid = np.bincount(nearest, weights=weighted_power, minlength=size)
        sums += factor * (ifft(grid)[:max_mode + 1] * size)
        weighted_power *= offset
        factor *= increment / (r + 1)
    return sums


def r70_moments(degree, *, tol=1e-13, oversample=1.0):
    """E[T_k(beta)], k=0,...,degree, for the symmetric continuous R70 law.

    Tail integration, panel quadrature, and nonuniform Fourier summation
    target absolute error tol in exact arithmetic; floating-point roundoff
    is additional. For degree T>=2, setup costs
    O((T+log(1/tol))*log(T/tol)*log(1/tol)) operations and
    O((T+log(1/tol))*log(T/tol)) memory. If tol=T**(-s), these are
    O(s*(1+s)*(T+s*log(T))*log(T)**2) operations and
    O((1+s)*(T+s*log(T))*log(T)) memory. Constants are independent of
    T, tol, and s, with oversample fixed.
    These are moment errors, not relative errors in portfolio wealth.
    """
    degree = _moment_parameters(degree, tol, oversample)
    moments = np.zeros(degree + 1)
    moments[0] = 1
    if degree < 2:
        return moments
    phi, weight, tail = _r70_angular_quadrature(degree, tol, oversample)
    sums = _positive_fourier_sums(2 * phi, weight, degree // 2, tol / 16)
    moments[::2] = tail + sums.real
    moments[2::4] *= -1
    moments[0] = 1  # The zeroth moment and all odd moments are known exactly.
    return moments


def _moment_weights(moments):
    # w_j = (mu_0 + 2 sum_{k>0} mu_k cos(k theta_j))/N.
    coef = moments.copy()
    coef[0] /= np.sqrt(2)
    return idct(coef, type=2, norm="ortho") * np.sqrt(2 / len(coef))


@dataclass
class WealthResult:
    log_wealth: np.ndarray
    log_co96: np.ndarray
    log_r70: np.ndarray
    co96_allocation: np.ndarray
    r70_bet: np.ndarray
    clipped_allocations: int


@dataclass
class MeanTestResult:
    mu0: float
    alpha: float
    wealth: WealthResult
    p_values: np.ndarray
    stopping_time: int | None

    @property
    def rejected(self):
        """Whether the threshold was reached at any observed time."""
        return self.stopping_time is not None


class FFTPortfolio:
    """Reusable moments and FFT evaluator for a specified maximum horizon.

    For a fixed null mean, every allocation depends only on its prefix.
    Fix the horizon in advance when interpreting rounded allocations as a
    betting strategy. With exact moments and exact arithmetic the answer is
    horizon-independent.
    """

    def __init__(self, horizon, *, leaf_size=64, moment_tol=1e-13, moment_oversample=1.0):
        if int(horizon) != horizon or horizon < 0:
            raise ValueError("horizon must be a nonnegative integer")
        if int(leaf_size) != leaf_size or leaf_size < 1:
            raise ValueError("leaf_size must be a positive integer")
        self.horizon = int(horizon)
        self._tree_size = 1 << max(0, (max(1, self.horizon) - 1).bit_length())
        self.leaf_size = int(leaf_size)
        self.moments = r70_moments(self._tree_size + 1, tol=moment_tol,
                                   oversample=moment_oversample)
        self._initial_weights = {}
        for prior in ("co96", "r70"):
            length = self._tree_size
            z = _nodes(length + 1)
            coordinate = (1 + z) / 2 if prior == "co96" else z
            weight = (np.full(length + 1, 1 / (length + 1)) if prior == "co96"
                      else _moment_weights(self.moments[:length + 1]))
            functional = np.stack((weight, weight * coordinate))
            self._initial_weights[prior, length] = functional
            while length > self.leaf_size:
                length //= 2
                functional = _unit_scale(_restrict(functional, length + 1))
                self._initial_weights[prior, length] = functional

    def _allocations(self, x, mean, prior):
        n = len(x)
        if n == 0:
            return np.empty(0), 0
        # Skip unused ancestors using their cached, deterministically restricted
        # functionals. Thus requesting a shorter prefix never changes its bets.
        length = self._tree_size
        while length > self.leaf_size and n <= length // 2:
            length //= 2
        weights = self._initial_weights[prior, length]
        allocations = np.empty(n)
        clipped = 0

        def visit(left, right, functional):
            nonlocal clipped
            length = right - left
            if length <= self.leaf_size:
                # The extra node also represents the RETURNED degree-length
                # product. Functional values use prefixes before observing x[t].
                grid = _nodes(length + 1)
                p = (1 + grid) / 2 if prior == "co96" else grid
                stop = min(right, n)
                observed = x[left:stop, None]
                if prior == "co96":
                    # Multiplying all returns by mean*(1-mean) leaves the
                    # posterior unchanged and avoids large ratios near 0/1.
                    factors = p * observed * (1 - mean) + (1 - p) * (1 - observed) * mean
                else:
                    factors = 1 + p * (observed - mean)
                factors /= factors.max(axis=1, keepdims=True)
                if length <= 256:
                    # At the central grid node each normalized linear factor
                    # is >= 1/2; blocks of this size cannot underflow there.
                    products = np.vstack((np.ones(length + 1), np.cumprod(factors, axis=0)))
                    products /= products.max(axis=1, keepdims=True)
                else:
                    with np.errstate(divide="ignore"):
                        log_products = np.vstack((np.zeros(length + 1), np.cumsum(np.log(factors), axis=0)))
                    products = np.exp(log_products - log_products.max(axis=1, keepdims=True))
                # Row t contains only observations strictly before t. No
                # block-wide normalization using future returns is permitted.
                integrals = products[:-1] @ functional.T
                bad = (integrals[:, 0] <= 0) | ~np.all(np.isfinite(integrals), axis=1)
                allocation = np.divide(integrals[:, 1], integrals[:, 0],
                                       out=np.full(stop - left, mean if prior == "co96" else 0.0), where=~bad)
                low, high = (0.0, 1.0) if prior == "co96" else (-1.0, 1.0)
                clipped += int(np.sum(bad | (allocation < low) | (allocation > high)))
                allocations[left:stop] = np.clip(allocation, low, high)
                return products[-1] if stop == right else None
            middle = (left + right) // 2
            left_product = visit(left, middle, _unit_scale(_restrict(functional, middle - left + 1)))
            if left_product is None or middle >= n:
                return None
            left_values = _resample(left_product, length + 1)
            right_functional = _restrict(functional * left_values, right - middle + 1)
            right_product = visit(middle, right, _unit_scale(right_functional))
            if right_product is None:
                return None
            return _unit_scale(left_values * _resample(right_product, length + 1))

        visit(0, length, weights)
        return allocations, clipped

    def wealth(self, x, mean):
        """All prefix log wealths, with time-zero wealth 1 implicit.

        Component allocations are projected to their admissible intervals.
        Accumulating their actual returns avoids thresholding a possibly
        inaccurate signed quadrature sum. Hedge the two WEALTHS, not bounds.
        """
        x = _data(x)
        if len(x) > self.horizon:
            raise ValueError("data exceed the configured horizon")
        if not np.isfinite(mean) or not 0 < mean < 1:
            raise ValueError("null mean must be strictly between 0 and 1")
        p, clipped_co96 = self._allocations(x, mean, "co96")
        beta, clipped_r = self._allocations(x, mean, "r70")
        with np.errstate(divide="ignore"):
            log_co96 = np.cumsum(np.logaddexp(np.log(p) + np.log(x) - np.log(mean),
                                          np.log1p(-p) + np.log1p(-x) - np.log1p(-mean)))
            log_r = np.cumsum(np.log1p(beta * (x - mean)))
        return WealthResult(np.logaddexp(log_co96, log_r) - np.log(2), log_co96, log_r,
                            p, beta, clipped_co96 + clipped_r)


    def test(self, x, mu0, alpha=0.05):
        """Test E[X] = mu0 against E[X] != mu0, reusing the stored moments.

        stopping_time is the first crossing (one-based), or None if no
        crossing was observed. Later declines in wealth do not undo rejection.
        The p-values use running maximum wealth, including initial wealth 1.
        """
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1")
        wealth = self.wealth(x, mu0)
        crossings = np.flatnonzero(wealth.log_wealth >= -np.log(alpha))
        running_log_wealth = np.maximum.accumulate(np.maximum(0.0, wealth.log_wealth))
        return MeanTestResult(float(mu0), float(alpha), wealth,
                              np.exp(-running_log_wealth),
                              int(crossings[0]) + 1 if len(crossings) else None)


def betting_mean_test(x, mu0, alpha=0.05, *, leaf_size=64, moment_tol=1e-13):
    """Run OPT-FFT for a specified interior null mean and observations in [0, 1].

    This batch interface returns all prefix wealths, anytime p-values, and
    the first rejection time. For repeated tests, reuse FFTPortfolio.test.
    """
    x = _data(x)
    return FFTPortfolio(len(x), leaf_size=leaf_size, moment_tol=moment_tol).test(x, mu0, alpha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="one-column CSV/text file, or .npy array")
    parser.add_argument("--mu0", type=float, required=True, help="null mean, strictly between 0 and 1")
    parser.add_argument("--alpha", type=float, default=0.05, help="type-I error level")
    parser.add_argument("--moment-tol", type=float, default=1e-13,
                        help="absolute moment approximation tolerance (default: 1e-13)")
    parser.add_argument("--output", type=Path, default=Path("mean_test.csv"))
    args = parser.parse_args()
    x = np.load(args.input) if args.input.suffix == ".npy" else np.loadtxt(args.input, delimiter=",")
    x = np.atleast_1d(x)
    try:
        result = betting_mean_test(x, args.mu0, args.alpha, moment_tol=args.moment_tol)
    except ValueError as error:
        parser.error(str(error))
    times = np.arange(1, len(x) + 1)
    rejected = times >= result.stopping_time if result.rejected else np.zeros(len(x), dtype=bool)
    np.savetxt(args.output, np.column_stack((times, x, result.wealth.log_wealth, result.p_values, rejected)),
               delimiter=",", header="n,x,log_wealth,p_value,rejected", comments="",
               fmt=["%d", "%.17g", "%.17g", "%.17g", "%d"])
    if result.rejected:
        print(f"OPT-FFT: Rejected E[X] = {args.mu0:g} at observation {result.stopping_time} (alpha={args.alpha:g}).")
    else:
        print(f"OPT-FFT: Did not reject E[X] = {args.mu0:g} within {len(x)} observations (alpha={args.alpha:g}).")
    print(f"Saved the test trace to {args.output}")


if __name__ == "__main__":
    main()
