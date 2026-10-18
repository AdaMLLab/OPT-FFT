"""OPT-FFT: Optimal Portfolio Testing with Fast Fourier Transforms.

Test a fixed bounded mean against Bernoulli means above the null, with increasing difficulty.

The test mixes the CO96 (Cover and Ordentlich, 1996) and R70 (Robbins, 1970)
priors, following Orabona and Jun (2024), with equal weights.

Run: OPENBLAS_NUM_THREADS=1 python demo.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/optimal_bet_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/optimal_bet_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from opt_fft import FFTPortfolio, r70_quadrature

ROOT = Path(__file__).resolve().parent


def direct_wealth(x, mu0):
    """Positive-quadrature reference used only for numerical validation."""
    count = (len(x) + 2) // 2
    p = (1 + np.cos(np.pi * (np.arange(count) + .5) / count)) / 2
    co96_weights = np.full(count, 1 / count)
    beta, weights, tail = r70_quadrature(len(x) + 1, oversample=1.5)
    beta = np.r_[beta, -beta, 0]
    r_weights = np.r_[weights / 2, weights / 2, tail]
    co96, r = np.empty(len(x)), np.empty(len(x))
    log_co96, log_r = 0.0, 0.0
    for i, value in enumerate(x):
        co96_weights *= p * value / mu0 + (1 - p) * (1 - value) / (1 - mu0)
        total = co96_weights.sum()
        log_co96 += np.log(total)
        co96_weights /= total
        r_weights *= 1 + beta * (value - mu0)
        total = r_weights.sum()
        log_r += np.log(total)
        r_weights /= total
        co96[i], r[i] = log_co96, log_r
    return np.logaddexp(co96, r) - np.log(2)


def plot_tests(scenarios, traces, mu0, alpha, horizon, trials, out):
    display_ceiling = -1.1 * np.log(alpha)
    gaps = sorted({row["gap"] for row in scenarios}, reverse=True)
    columns = min(2, len(gaps))
    trace_rows = (len(gaps) + columns - 1) // columns
    figure = plt.figure(figsize=(12, 3 * trace_rows + 3.2), constrained_layout=True)
    grid = figure.add_gridspec(trace_rows + 1, columns)
    time_axis = figure.add_subplot(grid[-1, :])
    for index, (row, (mean_log_wealth, se_log_wealth)) in enumerate(zip(scenarios, traces)):
        axis = figure.add_subplot(grid[index // columns, index % columns])
        # Crop the display only; every average still uses every trial.
        crossings = np.flatnonzero(mean_log_wealth >= display_ceiling)
        length = int(crossings[0]) + 1 if len(crossings) else len(mean_log_wealth)
        mean_log_wealth = mean_log_wealth[:length]
        se_log_wealth = se_log_wealth[:length]
        times = np.arange(1, length + 1)
        axis.plot(times, mean_log_wealth, color="C1", linewidth=1.3,
                  label=f"mean = {row['true_mean']:g}")
        if trials > 1:
            axis.fill_between(times, mean_log_wealth - se_log_wealth,
                              mean_log_wealth + se_log_wealth, color="C1", alpha=.18)
        axis.axhline(-np.log(alpha), color="black", linestyle="--", linewidth=1,
                     label="Individual-test threshold")
        axis.set(title=f"Mean gap above the null = {row['gap']:g}", xlabel="Observations",
                 ylabel="Average log wealth", xscale="log", xlim=(1, max(2, length)))
        axis.set_ylim(top=display_ceiling)
        axis.grid(True, alpha=.25)
        axis.legend(fontsize=8)
    time_axis.errorbar([row["gap"] for row in scenarios], [row["mean_capped_time"] for row in scenarios],
                       yerr=[row["se_capped_time"] for row in scenarios], marker="s", color="C1",
                       capsize=3, label="Mean above the null")
    time_axis.set(xlabel="Mean gap above the null (difficulty increases to the right)",
                  ylabel=r"Mean $\min(\tau, N)$", xscale="log", yscale="log",
                  xlim=(max(gaps) * 1.15, min(gaps) / 1.15),
                  title=f"{trials} trials per alternative; N = {horizon:,}; bars show one standard error")
    time_axis.grid(True, alpha=.25)
    time_axis.legend()
    figure.suptitle(f"OPT-FFT (CO96/R70): testing E[X] = {mu0:g}, alpha = {alpha:g}\n"
                   f"Average log wealth over {trials} trials; all paths continue after rejection"
                   + ("; shading: ±1 SE" if trials > 1 else ""), fontsize=12)
    figure.savefig(out / "mean_tests.png", dpi=180)
    figure.savefig(out / "mean_tests.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10000, help="maximum observations per trial")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mu0", type=float, default=.5)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--gaps", type=float, nargs="+", default=[.2, .1, .05, .025],
                        help="positive mean gaps above mu0, each smaller than 1-mu0")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--validate", action="store_true", help="check the first 256 observations per scenario against positive quadrature")
    args = parser.parse_args()
    if args.n < 1 or args.trials < 1:
        parser.error("--n and --trials must be positive")
    if not np.isfinite(args.mu0) or not 0 < args.mu0 < 1:
        parser.error("--mu0 must be strictly between 0 and 1")
    if not np.isfinite(args.alpha) or not 0 < args.alpha < 1:
        parser.error("--alpha must be strictly between 0 and 1")
    if any(not np.isfinite(gap) or not 0 < gap < 1 - args.mu0 for gap in args.gaps):
        parser.error("each gap must be positive and smaller than 1-mu0")
    gaps = sorted(set(args.gaps), reverse=True)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    engine = FFTPortfolio(args.n)
    setup_seconds = perf_counter() - start
    # Common random numbers make scenarios reproducible and comparable;
    # the trials within each scenario are independent.
    uniforms = np.random.default_rng(args.seed).uniform(size=(args.trials, args.n))
    scenarios, traces, trial_rows = [], [], []
    print(f"OPT-FFT: testing E[X] = {args.mu0:g}; alpha={args.alpha:g}, horizon={args.n}, trials={args.trials}, seed={args.seed}", flush=True)
    print(f"Moment setup: {setup_seconds:.3f}s", flush=True)
    for gap in gaps:
        true_mean = args.mu0 + gap
        capped, rejected, clipping = [], [], 0
        mean_log_wealth = np.zeros(args.n)
        sum_squared_deviations = np.zeros(args.n)
        start = perf_counter()
        for trial in range(args.trials):
            x = (uniforms[trial] < true_mean).astype(float)
            # Every trial contributes at every time, including after rejection.
            # Average LOG wealth, not wealth followed by a logarithm.
            result = engine.test(x, args.mu0, args.alpha)
            delta = result.wealth.log_wealth - mean_log_wealth
            mean_log_wealth += delta / (trial + 1)
            sum_squared_deviations += delta * (result.wealth.log_wealth - mean_log_wealth)
            observed_time = result.stopping_time if result.rejected else args.n
            capped.append(observed_time)
            rejected.append(result.rejected)
            clipping += result.wealth.clipped_allocations
            trial_rows.append({"true_mean": true_mean, "gap": gap, "side": "above",
                               "trial": trial + 1, "rejected": result.rejected,
                               "stopping_time": result.stopping_time, "capped_time": observed_time})
        seconds = perf_counter() - start
        se_log_wealth = (np.sqrt(sum_squared_deviations / ((args.trials - 1) * args.trials))
                         if args.trials > 1 else np.zeros(args.n))
        traces.append((mean_log_wealth, se_log_wealth))
        name = f"trace_gap_{gap:g}_above.csv"
        np.savetxt(out / name,
                   np.column_stack((np.arange(1, args.n + 1), mean_log_wealth,
                                    se_log_wealth, np.full(args.n, args.trials))),
                   delimiter=",", header="n,mean_log_wealth,se_log_wealth,trials", comments="",
                   fmt=["%d", "%.17g", "%.17g", "%d"])
        count = sum(rejected)
        row = {"true_mean": true_mean, "gap": gap, "side": "above",
               "rejections": count, "trials": args.trials, "rejection_rate": count / args.trials,
               "censored_trials": args.trials - count, "mean_capped_time": float(np.mean(capped)),
               "se_capped_time": float(np.std(capped, ddof=1) / np.sqrt(args.trials)) if args.trials > 1 else 0.0,
               "trace_file": name, "test_seconds": seconds,
               "clipped_allocations_in_full_traces": clipping}
        if args.validate:
            x = (uniforms[0, :min(256, args.n)] < true_mean).astype(float)
            fast = engine.wealth(x, args.mu0)
            error = float(np.max(abs(fast.log_wealth - direct_wealth(x, args.mu0))))
            if error >= 1e-8:
                raise RuntimeError(f"FFT/quadrature validation failed for mean {true_mean:g}: {error:g}")
            row["validation"] = {"observations": len(x), "max_log_wealth_error": error}
        scenarios.append(row)
        print(f"mean={true_mean:.4f}, gap={gap:g}: rejected {count}/{args.trials}, "
              f"mean min(tau,N)={row['mean_capped_time']:.1f}, censored={args.trials-count}, {seconds:.3f}s", flush=True)
    with (out / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trial_rows[0]))
        writer.writeheader()
        writer.writerows(trial_rows)
    summary_fields = ["true_mean", "gap", "side", "trials", "rejections", "rejection_rate",
                      "censored_trials", "mean_capped_time", "se_capped_time"]
    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scenarios)
    report = {"method": "OPT-FFT", "prior_names": ["CO96", "R70"], "distribution": "Bernoulli", "mu0": args.mu0,
              "alpha": args.alpha, "horizon": args.n, "trials": args.trials, "gaps": gaps,
              "seed": args.seed, "prior_weights": [.5, .5], "moment_setup_seconds": setup_seconds,
              "trace_statistic": "sample mean of log wealth over all trials at each time, continuing after rejection",
              "trace_uncertainty": "pointwise standard error across trials; zero placeholder for a single trial",
              "time_statistic": "sample mean of min(stopping time, horizon); unrejected trials are censored",
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "platform": platform.platform()},
              "scenarios": scenarios}
    (out / "mean_tests.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot_tests(scenarios, traces, args.mu0, args.alpha, args.n, args.trials, out)
    print(f"Saved test traces, trial results, summary, and plots in {out}", flush=True)


if __name__ == "__main__":
    main()
