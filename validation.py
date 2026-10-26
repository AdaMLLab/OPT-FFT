"""Direct positive-grid reference for tests and benchmarks, not a test engine."""

import numpy as np
from scipy.special import logsumexp

from opt_fft import WealthResult, _nodes, _positive_integer
from prior_moments import _legendre_rule



def fixed_prior_grids(co96_nodes=512, r70_order=16):
    """Return positive CO96 and R70 quadrature rules fixed before seeing data.

    CO96 uses a Gauss--Chebyshev rule, exact for polynomials of degree at most
    2*co96_nodes-1 in exact arithmetic. R70 uses Gauss--Legendre panels in
    y=-log(abs(beta)), with each panel normalized to its analytic prior mass.
    The mass at abs(beta)<exp(-32) is placed at beta=0. Its grid accuracy must
    be assessed separately; r70_order is not an absolute error tolerance.
    """
    co96_nodes = _positive_integer(co96_nodes, "co96_nodes")
    r70_order = _positive_integer(r70_order, "r70_order")
    p = (1 + _nodes(co96_nodes)) / 2
    co96 = (p, np.full(co96_nodes, 1 / co96_nodes))

    # Resolve endpoint concentration at |beta|=1, moderate bets, and small bets.
    # This partition and both quadrature orders are independent of the data.
    edges = np.unique(np.r_[0., 2. ** np.arange(-10, -2),
                            np.arange(.125, 8.125, .125),
                            np.arange(8.5, 16.5, .5), np.arange(17., 33.)])
    z, weight = _legendre_rule(r70_order)
    half_width = np.diff(edges)[:, None] / 2
    y = edges[:-1, None] + (z + 1) * half_width
    a = 1 + np.log(6.6)
    normalizer = np.log(a)
    weights = half_width * weight * normalizer / ((a + y) * np.log(a + y) ** 2)
    # Stable evaluation of the difference of reciprocal logarithms.
    panel_mass = (normalizer * np.log1p(np.diff(edges) / (a + edges[:-1]))
                  / (np.log(a + edges[:-1]) * np.log(a + edges[1:])))
    weights *= panel_mass[:, None] / weights.sum(axis=1)[:, None]
    beta = np.exp(-y.ravel())
    weights = weights.ravel()
    r70_weights = np.r_[weights / 2, normalizer / np.log(a + edges[-1]), weights / 2]
    r70_weights /= r70_weights.sum()
    r70 = (np.r_[-beta, 0., beta], r70_weights)
    for coordinate, prior_weight in (co96, r70):
        coordinate.setflags(write=False)
        prior_weight.setflags(write=False)
    return co96, r70


def direct_grid_wealth(x, mean, *, co96_nodes=512, r70_order=16):
    """Evaluate prefix mixture sums independently of the FFT/block recursion.

    Each prefix is integrated using log-sum-exp. Refining these grids checks
    numerical convergence empirically; it does not certify continuous-prior
    approximation error.
    """
    x = np.asarray(x, dtype=float)
    components, allocations = [], []
    for prior, (coordinate, weight) in zip(("co96", "r70"), fixed_prior_grids(co96_nodes, r70_order)):
        logs = np.log(weight)
        wealth, bets = np.empty(len(x)), np.empty(len(x))
        for t, value in enumerate(x):
            posterior = np.exp(logs - logsumexp(logs))
            bets[t] = posterior @ coordinate
            if prior == "co96":
                logs += np.log(coordinate*value/mean + (1-coordinate)*(1-value)/(1-mean))
            else:
                logs += np.log1p(coordinate*(value-mean))
            wealth[t] = logsumexp(logs)
        components.append(wealth)
        allocations.append(bets)
    co96, r70 = components
    return WealthResult(np.logaddexp(co96, r70)-np.log(2), co96, r70,
                        allocations[0], allocations[1], 0)
