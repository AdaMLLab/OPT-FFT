"""Benchmark the continuous-prior regional FFT implementation.

Run with OPENBLAS_NUM_THREADS=1 for repeatable small-array timings. Reported
times exclude engine construction, which is recorded separately. Error is
measured against a refined positive grid, with analytic CO96 checks on the
Bernoulli paths and an independent final R70 integral on reversal paths.
"""

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import betaln

from validation import direct_grid_wealth
from opt_fft import FFTPortfolio


def _number(value):
    return float(value) if np.isfinite(value) else str(float(value))


def _timed_wealth(engine, x, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = engine.wealth(x, .5)
        times.append(time.perf_counter() - start)
    return result, float(np.median(times))


def _reversal_reference(n, gap=.5):
    a = 1 + np.log(6.6)
    def integrand(y):
        return (np.expm1((n // 2) * np.log1p(-np.exp(-2*y)*gap**2))
                * np.log(a) / ((a+y)*np.log(a+y)**2))
    r70 = np.log1p(quad(integrand, 0, 64, epsabs=1e-13, epsrel=1e-13)[0])
    if gap == .5:
        co96 = betaln(n/2+.5, n/2+.5)-betaln(.5,.5)+n*np.log(2)
    else:
        co96 = np.log(2/np.pi*quad(
            lambda theta: np.exp((n//2)*np.log1p(-4*gap**2*np.cos(theta)**2)),
            0, np.pi/2, epsabs=1e-13, epsrel=1e-13)[0])
    return float(np.logaddexp(co96, r70)-np.log(2))


def run(lengths, repeats):
    rows = []
    for n in lengths:
        rng = np.random.default_rng(72)
        cases = {
            "reversal": np.r_[np.ones(n//2), np.zeros(n//2)],
            "bounded_reversal": np.r_[np.full(n//2, .9), np.full(n//2, .1)],
            "bernoulli_near_null": rng.binomial(1, .505, n),
            "bernoulli_strong": rng.binomial(1, .7, n),
            "bounded_beta": rng.beta(4, 3, n),
        }
        engines = {}
        setup = {}
        start = time.perf_counter()
        engines["opt_fft"] = FFTPortfolio(n).prepare(.5)
        setup["opt_fft"] = time.perf_counter()-start
        for case, x in cases.items():
            reference = direct_grid_wealth(x, .5, co96_nodes=max(1024, (n+2)//2), r70_order=32)
            analytic = None
            if np.all((x == 0) | (x == 1)):
                t = np.arange(1,n+1)
                s = np.cumsum(x)
                analytic = betaln(s+.5,t-s+.5)-betaln(.5,.5)+t*np.log(2)
            for method, engine in engines.items():
                result, seconds = _timed_wealth(engine,x,repeats)
                row = dict(n=n,case=case,method=method,
                           setup_seconds=setup[method],median_evaluation_seconds=seconds,
                           max_log_wealth_error=_number(np.max(abs(result.log_wealth-reference.log_wealth))),
                           final_log_wealth=_number(result.log_wealth[-1]),
                           clipped_allocations=result.clipped_allocations)
                if analytic is not None:
                    row["max_co96_log_error_analytic"] = _number(np.max(abs(result.log_co96-analytic)))
                if case in ("reversal", "bounded_reversal"):
                    row["final_log_error_independent"] = _number(abs(
                        result.log_wealth[-1]-_reversal_reference(n, .5 if case == "reversal" else .4)))
                row["regions_co96"] = result.regions_co96
                row["regions_r70"] = result.regions_r70
                row["panels_per_side"] = engine.panels_per_side
                rows.append(row)
                print(json.dumps(row,allow_nan=False),flush=True)
    return dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                repeats=repeats,implementation="regional_fft",leaf_size=32,moment_tol=1e-13,
                moment_oversample=1.,panels_per_side="polylogarithmic default",
                reference="Refined positive grid; independent checks as described in this script.",
                results=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths",type=int,nargs="+",default=[512,2048,8192])
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--output",type=Path,default=Path("results/benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 2 or n % 2 for n in args.lengths):
        parser.error("repeats must be positive and lengths must be positive even integers")
    report = run(args.lengths,args.repeats)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
