"""OPT-FFT: Optimal Portfolio Testing with Fast Fourier Transforms.

Test a specified bounded mean using the 50/50 CO96/R70 hedge, as in Remark 2 in Orabona and Jun (2024).

CO96 denotes the Beta(1/2, 1/2) portfolio prior analyzed by Cover and Ordentlich
(1996). R70 denotes the Robbins (1970) prior adapted by Orabona and Jun (2024)
to betting fractions in [-1, 1].

A fixed partition of the betting interval separates regions with very
unequal wealth. Each region uses the continuous conditional prior and its
own FFT-based computation of the wealth.

The rules are polynomial-exact through the configured horizon when their prior moments and arithmetic are
exact.

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
from scipy.fft import dct, idct

from prior_moments import panel_moments, prior_mass



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


def _positive_integer(value, name):
    if not np.isfinite(value) or value < 1 or int(value) != value:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _moment_weights(moments):
    coef = moments.copy()
    coef[..., 0] /= np.sqrt(2)
    return idct(coef, type=2, norm="ortho", axis=-1) * np.sqrt(2 / coef.shape[-1])


def _region_scale(values):
    # Every region has its own scale; never mix scales between regions.
    maximum = np.max(np.abs(values), axis=tuple(range(1, values.ndim)), keepdims=True)
    if np.any(~np.isfinite(maximum) | (maximum == 0)):
        raise FloatingPointError("degenerate polynomial/functional in a regional FFT")
    return values / maximum


def _default_panels(horizon):
    levels = max(1, int(np.ceil(np.log2(horizon + 1))))
    rounded = 1 << (levels * levels - 1).bit_length()
    tree = 1 << max(0, (max(1, horizon) - 1).bit_length())
    return min(rounded, max(1, tree // 16))



@dataclass
class WealthResult:
    log_wealth: np.ndarray
    log_co96: np.ndarray
    log_r70: np.ndarray
    co96_allocation: np.ndarray
    r70_bet: np.ndarray
    clipped_allocations: int
    regions_co96: int = 0
    regions_r70: int = 0



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
    """Regional continuous-prior FFT evaluator for a fixed maximum horizon.

    panels_per_side determines the partition, not the quadrature degree.
    Every region's integration rule covers the full horizon. A custom
    moment_provider(prior, lower, upper, degree) returns the normalized
    conditional Chebyshev moments m_0,...,m_degree of the continuous prior.
    The default provider evaluates them numerically. Fix the horizon and
    parameters before observing data to keep rounded allocations predictable.
    """

    def __init__(self, horizon, *, panels_per_side=None, leaf_size=32,
                 moment_tol=1e-13, moment_oversample=1.0, moment_provider=None):
        if not np.isfinite(horizon) or horizon < 0 or int(horizon) != horizon:
            raise ValueError("horizon must be a nonnegative integer")
        if not np.isfinite(moment_tol) or not 0 < moment_tol < .5:
            raise ValueError("moment tolerance must be strictly between 0 and 0.5")
        if not np.isfinite(moment_oversample) or moment_oversample < 1:
            raise ValueError("moment oversample must be finite and at least 1")
        if moment_provider is not None and not callable(moment_provider):
            raise ValueError("moment_provider must be callable")
        self.horizon = int(horizon)
        self._tree_size = 1 << max(0, (max(1, self.horizon) - 1).bit_length())
        self.panels_per_side = (_default_panels(self.horizon) if panels_per_side is None
                                else _positive_integer(panels_per_side, "panels_per_side"))
        self.leaf_size = _positive_integer(leaf_size, "leaf_size")
        self.moment_tol = float(moment_tol)
        self.moment_oversample = float(moment_oversample)
        self.moment_provider = moment_provider
        self._r70 = None
        self._co96 = None
        self._co96_mean = None

    @staticmethod
    def _check_mean(mean):
        if not np.isfinite(mean) or not 0 < mean < 1:
            raise ValueError("null mean must be strictly between 0 and 1")

    def _regions(self, prior, mean):
        # Cluster near the zero bet and the admissible endpoints. These
        # boundaries depend only on the configured horizon and candidate mean.
        fractions = np.sin(np.pi * np.arange(self.panels_per_side + 1)
                           / (2 * self.panels_per_side))**2
        fractions[0], fractions[-1] = 0., 1.
        if prior == "r70":
            regions = [(-float(fractions[1]), float(fractions[1]))]
            for lower, upper in zip(fractions[1:-1], fractions[2:]):
                regions.extend(((float(lower), float(upper)), (-float(upper), -float(lower))))
            return regions
        right = np.unique(mean + (1-mean)*fractions)
        if mean == .5:
            regions = []
            for lower, upper in zip(right[:-1], right[1:]):
                regions.extend(((float(lower), float(upper)), (float(1-upper), float(1-lower))))
            return regions
        left = np.unique(mean*(1-fractions[::-1]))
        return (list(zip(map(float, right[:-1]), map(float, right[1:])))
                + list(zip(map(float, left[:-1]), map(float, left[1:]))))

    def _prepare_component(self, prior, mean):
        regions = self._regions(prior, mean)
        length = self._tree_size
        z = _nodes(length + 1)
        root = np.empty((len(regions), 2, length + 1))
        masses = np.empty(len(regions))
        seen = {}
        for j, (lower, upper) in enumerate(regions):
            masses[j] = prior_mass(prior, lower, upper)
            reflection = ((1-upper, 1-lower) if prior == "co96" else (-upper, -lower))
            if reflection in seen:
                weights = root[seen[reflection], 0, ::-1]
            else:
                moments = (panel_moments(prior, lower, upper, length, tol=self.moment_tol,
                                         oversample=self.moment_oversample)
                           if self.moment_provider is None else
                           np.asarray(self.moment_provider(prior, lower, upper, length), dtype=float))
                if (moments.shape != (length + 1,) or not np.all(np.isfinite(moments))
                        or not np.isclose(moments[0], 1, rtol=0, atol=1e-12)):
                    raise ValueError("moment_provider must return degree+1 finite normalized moments")
                weights = _moment_weights(moments)
            root[j, 0] = weights
            # Compute a mean in the local coordinate z, then map it to the
            # betting interval. This avoids subtracting nearly equal large
            # numerator/denominator contributions on narrow regions.
            root[j, 1] = weights*z
            seen[lower, upper] = j
        masses /= masses.sum()
        initial = {length: root}
        while length > self.leaf_size:
            length //= 2
            root = _region_scale(_restrict(root, length + 1))
            initial[length] = root
        bounds = np.asarray(regions)
        return dict(initial=initial, masses=masses,
                    middle=bounds.mean(axis=1), half=np.diff(bounds, axis=1)[:, 0]/2)

    def prepare(self, mean):
        """Precompute moments/rules; useful for separating setup from timing.

        R70 is reusable across means. Only the most recent CO96 partition is
        retained, so testing many candidate means does not grow memory forever.
        """
        self._check_mean(mean)
        if self._r70 is None:
            self._r70 = self._prepare_component("r70", mean)
        if self._co96_mean != mean:
            self._co96 = None
            self._co96 = self._prepare_component("co96", mean)
            self._co96_mean = float(mean)
        return self

    def _component(self, x, mean, prior, prepared):
        middle, half = prepared["middle"], prepared["half"]
        count = len(middle)
        allocation = np.empty((count, len(x)))
        clipped = 0
        length = self._tree_size
        while length > self.leaf_size and len(x) <= length // 2:
            length //= 2

        def visit(left, right, functional):
            nonlocal clipped
            width = right-left
            if width <= self.leaf_size:
                z = _nodes(width + 1)
                coordinate = middle[:, None, None] + half[:, None, None]*z
                stop = min(right, len(x))
                observed = x[None, left:stop, None]
                factors = (coordinate*observed*(1-mean) + (1-coordinate)*(1-observed)*mean
                           if prior == "co96" else 1 + coordinate*(observed-mean))
                factors /= factors.max(axis=-1, keepdims=True)
                if width <= 256:
                    products = np.concatenate((np.ones((count, 1, width+1)),
                                               np.cumprod(factors, axis=1)), axis=1)
                    products /= products.max(axis=-1, keepdims=True)
                else:
                    logs = np.concatenate((np.zeros((count, 1, width+1)),
                                           np.cumsum(np.log(factors), axis=1)), axis=1)
                    products = np.exp(logs-logs.max(axis=-1, keepdims=True))
                integrals = products[:, :-1] @ functional.swapaxes(-1, -2)
                bad = (integrals[:, :, 0] <= 0) | ~np.all(np.isfinite(integrals), axis=-1)
                local = np.divide(integrals[:, :, 1], integrals[:, :, 0],
                                  out=np.zeros((count, stop-left)), where=~bad)
                clipped += int(np.sum(bad | (local < -1) | (local > 1)))
                allocation[:, left:stop] = middle[:, None] + half[:, None]*np.clip(local, -1, 1)
                return products[:, -1] if stop == right else None
            midpoint = (left+right)//2
            first = visit(left, midpoint, _region_scale(_restrict(functional, midpoint-left+1)))
            if first is None or midpoint >= len(x):
                return None
            values = _resample(first, width+1)
            second = visit(midpoint, right, _region_scale(
                _restrict(functional*values[:, None, :], right-midpoint+1)))
            if second is None:
                return None
            return _region_scale(values*_resample(second, width+1))

        visit(0, length, prepared["initial"][length])
        if prior == "co96":
            with np.errstate(divide="ignore"):
                increments = np.logaddexp(np.log(allocation)+np.log(x)-np.log(mean),
                                         np.log1p(-allocation)+np.log1p(-x)-np.log1p(-mean))
        else:
            increments = np.log1p(allocation*(x-mean))
        # Hedge the actual regional strategies. Only their PAST wealth enters
        # the allocation for the next observation; there is no retrospective
        # replacement of capital by a recomputed mixture value.
        logs = np.cumsum(increments, axis=1)
        previous = np.column_stack((np.zeros(count), logs[:, :-1]))
        previous += np.log(prepared["masses"])[:, None]
        weights = np.exp(previous-previous.max(axis=0))
        combined = np.sum(weights*allocation, axis=0)/weights.sum(axis=0)
        return np.clip(combined, 0 if prior == "co96" else -1, 1), clipped

    def wealth(self, x, mean):
        x = _data(x)
        if len(x) > self.horizon:
            raise ValueError("data exceed the configured horizon")
        self._check_mean(mean)
        if len(x) == 0:
            return WealthResult(*(np.empty(0) for _ in range(5)), 0)
        self.prepare(mean)
        p, clipped_co = self._component(x, mean, "co96", self._co96)
        beta, clipped_r = self._component(x, mean, "r70", self._r70)
        co96, r70, total = self._log_returns(x, mean, p, beta)
        return WealthResult(total, co96, r70, p, beta, clipped_co+clipped_r,
                            len(self._co96["masses"]), len(self._r70["masses"]))


    def test(self, x, mu0, alpha=0.05):
        """Test E[X] = mu0 against E[X] != mu0, reusing continuous-prior moment rules.

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

    @staticmethod
    def _log_returns(x, mean, p, beta):
        with np.errstate(divide="ignore"):
            co96 = np.cumsum(np.logaddexp(np.log(p) + np.log(x) - np.log(mean),
                                         np.log1p(-p) + np.log1p(-x) - np.log1p(-mean)))
            r70 = np.cumsum(np.log1p(beta * (x - mean)))
        return co96, r70, np.logaddexp(co96, r70) - np.log(2)


def betting_mean_test(x, mu0, alpha=0.05, *, panels_per_side=None, leaf_size=32,
                      moment_tol=1e-13, moment_oversample=1.0, moment_provider=None):
    """Run the continuous-prior regional FFT test for one candidate mean."""
    x = _data(x)
    return FFTPortfolio(len(x), panels_per_side=panels_per_side, leaf_size=leaf_size,
                        moment_tol=moment_tol, moment_oversample=moment_oversample,
                        moment_provider=moment_provider).test(x, mu0, alpha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="one-column CSV/text file, or .npy array")
    parser.add_argument("--mu0", type=float, required=True, help="null mean, strictly between 0 and 1")
    parser.add_argument("--alpha", type=float, default=0.05, help="type-I error level")
    parser.add_argument("--panels-per-side", type=int, default=None,
                        help="spatial partition size; default grows polylogarithmically with the horizon")
    parser.add_argument("--leaf-size", type=int, default=32, help="direct recursion leaf size")
    parser.add_argument("--moment-tol", type=float, default=1e-13,
                        help="numerical moment accuracy target; not a wealth-error bound")
    parser.add_argument("--moment-oversample", type=float, default=1.,
                        help="increase moment quadrature order for refinement checks")
    parser.add_argument("--output", type=Path, default=Path("mean_test.csv"))
    args = parser.parse_args()
    x = np.load(args.input) if args.input.suffix == ".npy" else np.loadtxt(args.input, delimiter=",")
    x = np.atleast_1d(x)
    try:
        result = betting_mean_test(x, args.mu0, args.alpha, panels_per_side=args.panels_per_side,
                                   leaf_size=args.leaf_size, moment_tol=args.moment_tol,
                                   moment_oversample=args.moment_oversample)
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
