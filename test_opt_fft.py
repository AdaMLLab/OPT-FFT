"""OPT-FFT: grid accuracy, independent integrals, and predictable FFT blocks."""

from itertools import product
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.special import betaln, logsumexp

from opt_fft import (FFTPortfolio, _nodes, _resample, _restrict,
                     betting_mean_test)
from prior_moments import panel_moments, prior_mass
from validation import direct_grid_wealth



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
    def test_continuous_region_moments_and_masses(self):
        from scipy.special import eval_chebyt
        for prior, lower, upper in (("co96", 0., .3), ("co96", .2, .6),
                                     ("co96", .9, 1.), ("r70", .01, .2)):
            actual = panel_moments(prior, lower, upper, 64)
            mass = prior_mass(prior, lower, upper)
            a = 1+np.log(6.6)
            def density(v):
                if prior == "co96":
                    return 1/(np.pi*np.sqrt(v*(1-v)))
                d = a-np.log(v)
                return np.log(a)/(2*v*d*np.log(d)**2)
            for k in (0, 1, 2, 5, 16):
                expected = quad(lambda v: eval_chebyt(k, (2*v-lower-upper)/(upper-lower))*density(v),
                                lower, upper, epsabs=1e-12, epsrel=1e-12, limit=1000)[0]/mass
                assert_allclose(actual[k], expected, atol=3e-11, rtol=0)
            refined = panel_moments(prior, lower, upper, 64, tol=1e-14, oversample=2)
            assert_allclose(actual, refined, atol=3e-12, rtol=0)
        radius = .03
        central = panel_moments("r70", -radius, radius, 64)
        a = 1+np.log(6.6)-np.log(radius)
        for k in (2, 4, 16, 64):
            excess = quad(lambda y: -2*np.sin(k*np.arcsin(np.exp(-y))/2)**2
                          *np.log(a)/((a+y)*np.log(a+y)**2),
                          0, 80, epsabs=1e-13, epsrel=1e-13, limit=1000)[0]
            expected = (-1)**(k//2)*(1+excess)
            assert_allclose(central[k], expected, atol=3e-12, rtol=0)
        assert_array_equal(central[1::2], 0)
        engine = FFTPortfolio(100, panels_per_side=4)
        for mean in (.2, .5, .8):
            for prior in ("co96", "r70"):
                masses = [prior_mass(prior, lo, hi) for lo, hi in engine._regions(prior, mean)]
                assert_allclose(sum(masses), 1, atol=3e-15, rtol=0)
                self.assertTrue(all(m > 0 for m in masses))

    def test_supplied_moments_and_partition_invariance(self):
        x = np.random.default_rng(19).uniform(size=37)
        calls = []
        def supplied(prior, lower, upper, degree):
            calls.append((prior, lower, upper, degree))
            return panel_moments(prior, lower, upper, degree, tol=1e-14, oversample=2)
        for panels in (1, 2, 4):
            result = FFTPortfolio(len(x), panels_per_side=panels, leaf_size=4,
                                  moment_provider=supplied).wealth(x, .4)
            for n in (1, 8, 17, 37):
                assert_allclose(result.log_co96[n-1], direct_co96_log(x[:n], .4), atol=3e-11)
                assert_allclose(result.log_r70[n-1], np.log(direct_r70(x[:n], .4)), atol=3e-11)
        self.assertTrue(calls)
        self.assertTrue(all(degree >= len(x) for _, _, _, degree in calls))

    def test_cli_uses_the_default_engine_and_new_parameters(self):
        x = np.r_[np.ones(30), np.zeros(30)]
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary)/"data.npy", Path(temporary)/"trace.csv"
            np.save(source, x)
            command = [sys.executable, str(Path(__file__).with_name("opt_fft.py")),
                       str(source), "--mu0", ".4", "--output", str(output)]
            for options in ([], ["--panels-per-side", "4", "--leaf-size", "2",
                                 "--moment-tol", "1e-12", "--moment-oversample", "1.5"]):
                completed = subprocess.run(command+options, capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                parameters = (dict(panels_per_side=4, leaf_size=2, moment_tol=1e-12, moment_oversample=1.5) if options else {})
                expected = betting_mean_test(x, .4, **parameters)
                trace = np.loadtxt(output, delimiter=",", skiprows=1)
                assert_allclose(trace[:, 2], expected.wealth.log_wealth, atol=1e-13)
                assert_allclose(trace[:, 3], expected.p_values, atol=1e-13)
                assert_array_equal(trace[:, 4], np.arange(1, len(x)+1) >= expected.stopping_time)
            rejected = subprocess.run(command+["--panels-per-side", "0"], capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("positive integer", rejected.stderr)

    def test_invalid_data_and_test_parameters(self):
        for bad in ([np.nan], [-.1], [1.1], [[.5]], [np.inf]):
            with self.assertRaises(ValueError):
                betting_mean_test(bad, .5)
        for alpha in (0, 1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                betting_mean_test([.5], .5, alpha)
        for mean in (0, 1, -.1, 1.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                betting_mean_test([.5], mean)

    def test_grid_and_adjoint(self):
        rng = np.random.default_rng(8)
        for small, large in ((1, 7), (4, 15), (17, 35)):
            coefficients = rng.normal(size=small)
            values = np.polynomial.chebyshev.chebval(_nodes(small), coefficients)
            expected = np.polynomial.chebyshev.chebval(_nodes(large), coefficients)
            assert_allclose(_resample(values, large), expected, atol=2e-13)
            weights = rng.normal(size=large)
            assert_allclose(weights @ expected, _restrict(weights, small) @ values, atol=2e-13)

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

    def test_reversal_matches_independent_integrals(self):
        x = np.r_[np.ones(256), np.zeros(256)]
        result = FFTPortfolio(len(x)).wealth(x, .5)
        t = np.arange(1, len(x)+1)
        s = np.cumsum(x)
        co96 = betaln(s+.5, t-s+.5)-betaln(.5,.5)+t*np.log(2)
        assert_allclose(result.log_co96, co96, atol=2e-8, rtol=0)
        for end in (256, 260, 320, 512):
            assert_allclose(result.log_r70[end-1], np.log(direct_r70(x[:end], .5)), atol=2e-8, rtol=0)
        self.assertEqual(result.clipped_allocations, 0)
        self.assertGreater(result.regions_co96, 1)
        self.assertGreater(result.regions_r70, 1)

    def test_predictability_across_block_boundaries(self):
        x = np.r_[np.ones(96), np.zeros(96)]
        engine = FFTPortfolio(len(x), leaf_size=4)
        full = engine.wealth(x, .5)
        for end in (1, 7, 16, 33, 96, 97, 100, 128, 191):
            prefix = engine.wealth(x[:end], .5)
            changed = x.copy()
            changed[end-1:] = 1 - changed[end-1:]
            other = engine.wealth(changed, .5)
            assert_allclose(prefix.log_wealth, full.log_wealth[:end], atol=2e-12, rtol=0)
            assert_allclose(other.co96_allocation[:end], full.co96_allocation[:end], atol=2e-12, rtol=0)
            assert_allclose(other.r70_bet[:end], full.r70_bet[:end], atol=2e-12, rtol=0)

    def test_accumulated_returns_and_test_api(self):
        x = np.r_[np.ones(64), np.zeros(64)]
        engine = FFTPortfolio(len(x))
        test = engine.test(x, .4, .05)
        result = test.wealth
        co96 = np.cumsum(np.log(result.co96_allocation*x/.4
                               + (1-result.co96_allocation)*(1-x)/.6))
        r70 = np.cumsum(np.log1p(result.r70_bet*(x-.4)))
        assert_allclose(result.log_co96, co96, atol=2e-12, rtol=0)
        assert_allclose(result.log_r70, r70, atol=2e-12, rtol=0)
        assert_allclose(result.log_wealth, np.logaddexp(co96, r70)-np.log(2), atol=2e-12, rtol=0)
        crossing = np.flatnonzero(result.log_wealth >= -np.log(.05))[0]+1
        self.assertEqual(test.stopping_time, crossing)
        self.assertTrue(np.all(np.diff(test.p_values) <= 0))
        empty = FFTPortfolio(0).test([], .5)
        self.assertIsNone(empty.stopping_time)
        self.assertEqual(empty.wealth.regions_co96, 0)

    def test_null_error_by_path_enumeration(self):
        engine = FFTPortfolio(7, panels_per_side=2, leaf_size=2)
        for mean in (.3, .7):
            probability = 0.
            for path in product((0., 1.), repeat=7):
                if engine.test(path, mean, .2).rejected:
                    successes = sum(path)
                    probability += mean**successes*(1-mean)**(7-successes)
            self.assertLessEqual(probability, .2+1e-12)

    def test_invalid_parameters(self):
        for horizon in (-1, .5, np.inf, np.nan):
            with self.assertRaises(ValueError):
                FFTPortfolio(horizon)
        for name in ("panels_per_side", "leaf_size"):
            for value in (0, -1, .5, np.inf, np.nan):
                with self.assertRaises(ValueError):
                    FFTPortfolio(8, **{name: value})
        for tol in (0, -.1, .5, np.inf, np.nan):
            with self.assertRaises(ValueError):
                FFTPortfolio(8, moment_tol=tol)
        for oversample in (0, .5, np.inf, np.nan):
            with self.assertRaises(ValueError):
                FFTPortfolio(8, moment_oversample=oversample)
        with self.assertRaises(ValueError):
            FFTPortfolio(8, moment_provider=1)
        with self.assertRaises(ValueError):
            FFTPortfolio(8, moment_provider=lambda *args: np.ones(1)).wealth([.5], .5)
        engine = FFTPortfolio(1)
        for x, mean in (([0, 1], .5), ([np.nan], .5), ([.5], 0), ([.5], 1)):
            with self.assertRaises(ValueError):
                engine.wealth(x, mean)


if __name__ == "__main__":
    unittest.main()
