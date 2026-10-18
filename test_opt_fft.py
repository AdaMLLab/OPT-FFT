"""OPT-FFT checks: independent quadrature, analytic cases, and predictability."""

from itertools import product
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.special import betaln, logsumexp

from opt_fft import FFTPortfolio, _nodes, _resample, _restrict, betting_mean_test, r70_moments


def direct_r70(x, mean):
    """Adaptive scalar integration, independent of production moment code."""
    a = 1 + np.log(6.6)
    normalizer = np.log(a)

    def integrand(y):
        beta = np.exp(-y)
        # Integrate wealth minus its value at beta=0; the remaining mass is 1.
        log_plus = np.log1p(beta * (x - mean)).sum()
        log_minus = np.log1p(-beta * (x - mean)).sum()
        return (np.expm1(log_plus) + np.expm1(log_minus)) * normalizer / (2 * (a + y) * np.log(a + y) ** 2)

    return 1 + quad(integrand, 0, 80, epsabs=2e-12, epsrel=2e-12)[0]


def direct_co96_log(x, mean):
    size = (len(x) + 2) // 2
    p = (1 + _nodes(size)) / 2
    log_products = np.log(x[:, None] * p / mean + (1 - x[:, None]) * (1 - p) / (1 - mean)).sum(axis=0)
    return logsumexp(log_products) - np.log(size)


class FFTTests(unittest.TestCase):
    def test_grid_and_adjoint(self):
        rng = np.random.default_rng(8)
        for small, large in ((1, 7), (4, 15), (17, 35)):
            coefficients = rng.normal(size=small)
            values = np.polynomial.chebyshev.chebval(_nodes(small), coefficients)
            expected = np.polynomial.chebyshev.chebval(_nodes(large), coefficients)
            assert_allclose(_resample(values, large), expected, atol=2e-13)
            weights = rng.normal(size=large)
            assert_allclose(weights @ expected, _restrict(weights, small) @ values, atol=2e-13)

    def test_continuous_r70_moments(self):
        moments = r70_moments(70)
        a = 1 + np.log(6.6)
        for k in (2, 4, 16, 40, 70):
            center = (-1) ** (k // 2)
            def f(y):
                return (np.cos(k * np.arccos(np.exp(-y))) - center) * np.log(a) / ((a + y) * np.log(a + y) ** 2)
            expected = center + quad(f, 0, 40, epsabs=1e-12, limit=300)[0]
            self.assertAlmostEqual(moments[k], expected, places=11)
        assert_array_equal(moments[1::2], 0)

    def test_all_prefixes_against_independent_integrals(self):
        rng = np.random.default_rng(3)
        for x in (rng.uniform(size=37), rng.binomial(1, .4, size=37), np.full(37, .5)):
            engine = FFTPortfolio(len(x), leaf_size=8)
            for mean in (.25, .5, .8):
                result = engine.wealth(x, mean)
                for n in (1, 2, 7, 8, 9, 16, 17, 32, 37):
                    co96 = direct_co96_log(x[:n], mean)
                    r70 = np.log(direct_r70(x[:n], mean))
                    assert_allclose(result.log_co96[n - 1], co96, atol=2e-11)
                    assert_allclose(result.log_r70[n - 1], r70, atol=2e-11)
                    assert_allclose(result.log_wealth[n - 1], np.logaddexp(co96, r70) - np.log(2), atol=2e-11)

    def test_bernoulli_co96_formula(self):
        x = np.random.default_rng(71).binomial(1, .35, size=200)
        n, successes = np.arange(1, len(x) + 1), np.cumsum(x)
        mean = .36
        expected = betaln(successes + .5, n - successes + .5) - betaln(.5, .5) - successes * np.log(mean) - (n - successes) * np.log1p(-mean)
        assert_allclose(FFTPortfolio(len(x)).wealth(x, mean).log_co96, expected, atol=3e-12)

    def test_predictability_and_prefix_queries(self):
        x = np.random.default_rng(7).uniform(size=139)
        engine = FFTPortfolio(139, leaf_size=8)
        full = engine.wealth(x, .4)
        for n in (1, 7, 8, 9, 32, 83, 129):
            prefix = engine.wealth(x[:n], .4)
            assert_allclose(prefix.log_wealth, full.log_wealth[:n], atol=1e-13)
            changed = x.copy()
            changed[n - 1:] = 1 - changed[n - 1:]
            other = engine.wealth(changed, .4)
            assert_allclose(other.co96_allocation[:n], full.co96_allocation[:n], atol=1e-13)
            assert_allclose(other.r70_bet[:n], full.r70_bet[:n], atol=1e-13)

    def test_rejection_times_against_independent_integrals(self):
        for value, mu0 in ((1, .5), (0, .5), (.6, .5), (.5, .5)):
            x = np.full(37, value)
            alpha = .05
            result = betting_mean_test(x, mu0, alpha, leaf_size=8)
            reference = np.array([
                np.logaddexp(direct_co96_log(x[:n], mu0), np.log(direct_r70(x[:n], mu0))) - np.log(2)
                for n in range(1, len(x) + 1)
            ])
            hits = np.flatnonzero(reference >= -np.log(alpha))
            expected_time = int(hits[0]) + 1 if len(hits) else None
            self.assertEqual(result.stopping_time, expected_time)
            expected_p = np.minimum(1, np.exp(-np.maximum.accumulate(reference)))
            assert_allclose(result.p_values, expected_p, atol=2e-11)
            self.assertEqual(result.rejected, expected_time is not None)

    def test_first_crossing_is_not_undone(self):
        x = np.r_[np.ones(10), np.zeros(10)]
        result = betting_mean_test(x, .5, .05)
        self.assertTrue(result.rejected)
        self.assertLessEqual(result.stopping_time, 10)
        self.assertLess(result.wealth.log_wealth[-1], -np.log(.05))
        self.assertLessEqual(result.p_values[-1], .05)
        self.assertTrue(np.all(np.diff(result.p_values) <= 0))

    def test_two_sided_symmetry_and_empty_input(self):
        x = np.random.default_rng(5).binomial(1, .7, size=100)
        result = betting_mean_test(x, .4, .1)
        mirror = betting_mean_test(1 - x, .6, .1)
        assert_allclose(result.wealth.log_wealth, mirror.wealth.log_wealth, atol=2e-12)
        assert_allclose(result.p_values, mirror.p_values, atol=2e-12)
        self.assertEqual(result.stopping_time, mirror.stopping_time)
        empty = betting_mean_test([], .5, .05)
        self.assertFalse(empty.rejected)
        self.assertIsNone(empty.stopping_time)
        self.assertEqual(len(empty.p_values), 0)

    def test_null_error_by_enumerating_bernoulli_paths(self):
        # Sum the exact null probabilities of all paths that reject by n=8.
        engine, alpha = FFTPortfolio(8, leaf_size=4), .2
        for mu0 in (.2, .5, .8):
            probability = 0.0
            for path in product((0.0, 1.0), repeat=8):
                if engine.test(path, mu0, alpha).rejected:
                    successes = sum(path)
                    probability += mu0 ** successes * (1 - mu0) ** (8 - successes)
            self.assertLessEqual(probability, alpha + 1e-12)

    def test_reversal_keeps_bets_admissible(self):
        # Severe reversals may destroy FFT relative accuracy. Even then the
        # reported wealth must be the wealth of the clipped predictable bets.
        x = np.r_[np.ones(256), np.zeros(256)]
        result = FFTPortfolio(len(x)).wealth(x, .5)
        self.assertTrue(np.all((result.co96_allocation >= 0) & (result.co96_allocation <= 1)))
        self.assertTrue(np.all(abs(result.r70_bet) <= 1))
        with np.errstate(divide="ignore"):
            expected = np.cumsum(np.log(result.co96_allocation * x / .5 + (1 - result.co96_allocation) * (1 - x) / .5))
        assert_allclose(result.log_co96, expected)
        self.assertFalse(np.any(np.isnan(result.log_wealth)))

    def test_invalid_inputs(self):
        for bad in ([np.nan], [-.1], [1.1], [[.5]], [np.inf]):
            with self.assertRaises(ValueError):
                betting_mean_test(bad, .5)
        for alpha in (0, 1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                betting_mean_test([.5], .5, alpha)
        for mu0 in (0, 1, -.1, 1.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                betting_mean_test([.5], mu0)
        with self.assertRaises(ValueError):
            FFTPortfolio(1).wealth([.3, .4], .5)


if __name__ == "__main__":
    unittest.main()
