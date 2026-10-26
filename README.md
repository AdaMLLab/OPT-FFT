# OPT-FFT

**Optimal Portfolio Testing with Fast Fourier Transforms**

Test whether independent observations in `[0, 1]` have mean `mu0`, against the two-sided alternative that the mean differs from `mu0`. The implementation evaluates a 50/50 continuous mixture of the CO96 and R70 priors and rejects when betting wealth first reaches `1 / alpha`, as described in Remark 2 of Orabona and Jun (IEEE Trans. Information Theory, 2024).
Here, we denote by

- **CO96**: the `Beta(1/2, 1/2)` prior on the two-stock portfolio allocation analyzed by Cover and Ordentlich (1996).
- **R70**: the prior proposed by Robbins (1970), adapted by Orabona and Jun (2024) to betting fractions in `[-1, 1]`.

OPT-FFT combines a spatial grid with FFT recursion within each region. It evaluates the continuous-prior portfolio exactly in exact arithmetic, given exact prior moments. 

The algorithm and motivation are described in 

```
@article{orabon2026optimal,
  title={Optimal Priors for Betting Tests with Bounded Payoffs: An Exact Expected-Time Characterization},
  author={Orabona, Francesco},
  journal={arXiv preprint arXiv:2610.25821},
  year={2026}
}
```

## Installation

Run all commands from the `python/` directory:

```bash
python -m pip install -r requirements.txt
```

## Test a specified mean

```python
import numpy as np
from opt_fft import betting_mean_test

x = np.random.default_rng(2026).binomial(1, 0.6, size=10000)
result = betting_mean_test(x, mu0=0.5, alpha=0.05)
print(result.rejected, result.stopping_time)
```

Pass a finite one-dimensional array in `[0, 1]` and a null mean strictly between
zero and one. `alpha` is the type-I error level, also strictly between zero and
one. For data in `[a, b]`, use `(x - a) / (b - a)` and
`(mu0 - a) / (b - a)`.

The result contains:

- `rejected`: whether the test rejected at any observed time.
- `stopping_time`: the first rejection time, counted from one, or `None`.
- `p_values`: an array of anytime p-values.
- `wealth.log_wealth`: an array of log wealth at each observation.

Rejection remains recorded even if wealth later falls. Non-rejection does not
establish that the null is true.

## Command line

Supply a one-column CSV/text file without a header, or a `.npy` array:

```bash
python opt_fft.py data.npy --mu0 0.5 --alpha 0.05 --output mean_test.csv
```

The output columns are `n,x,log_wealth,p_value,rejected`. The rejection column
stays one after the first crossing.

## Regional grid and FFT evaluation

The default `FFTPortfolio` engine uses a spatial grid and FFT together:

1. Partition the allocation domain into regions, keeping the continuous
   conditional prior inside every region.
2. Turn each region's Chebyshev moments into an integration rule covering
   the full configured horizon. Evaluate prefix integrals by FFT recursion.
3. Compute allocations in each region's local coordinate, scale regions
   independently, and combine them using their preceding wealths and prior masses.

Given exact moments and exact arithmetic, every required polynomial integral
is exact, and the regional strategies recombine into the original continuous
mixture. The spatial boundaries do not define an atomic approximation to the
prior. The built-in moment provider computes numerical approximations to the
continuous prior's moments.

Long one-sided histories can make some contributions tiny before they become
important after a reversal. Independent regional scaling and logarithmic
wealth storage help retain them. Local coordinates improve the conditioning
of allocation calculations on narrow regions. The grid clusters near zero
bet and the support endpoints. R70's singularity at zero gets its own symmetric
region, analytic mass, and specialized moment calculation. These mechanisms
improve the accuracy in our experiments.

For repeated tests, reuse the precomputed moment rules:

```python
from opt_fft import FFTPortfolio

engine = FFTPortfolio(10000, leaf_size=32, moment_tol=1e-13).prepare(0.5)
result = engine.test(x, mu0=0.5, alpha=0.05)
print(result.wealth.regions_co96, result.wealth.regions_r70)
```

`prepare(mu0)` is optional: the first evaluation calls it automatically.
R70 moments can be reused across candidate means; the CO96 partition depends
on `mu0`, and only its most recently prepared rules are retained.
The same keyword parameters are accepted by `betting_mean_test`. CLI options
use hyphens, for example `--panels-per-side 128 --moment-oversample 2`.

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `panels_per_side` | horizon dependent | Number P controlling the spatial partition |
| `leaf_size` | 32 | Largest recursion leaf evaluated by direct products |
| `moment_tol` | `1e-13` | Moment accuracy target; not a certified wealth-error bound |
| `moment_oversample` | 1 | Multiplier for Gaussian orders; increase for refinement |
| `moment_provider` | built-in numerical routine | Optional provider of continuous conditional moments (Python API) |

For horizon T, let D be its next power of two and L = ceil(log2(T+1)). The default choice of P is the smaller of the next power of 2 at least L² and max(1,D/16). There are 2P CO96 regions and 2P−1 R70 regions.
At horizons 512, 2,048, and 8,192, P is 32, 128, and 256, respectively. Boundaries use squared-sine spacing; the central R70 interval joins both signs to exploit symmetry. Region count improves spatial resolution, while the moment degree in every region always covers the full horizon. Leaf size 32 avoids transform overhead on small products. Fix the horizon and numerical parameters before observing data.

For K total regions and T equal to the configured horizon, evaluation takes `O(K T log²(T+1))` arithmetic operations and `O(K T)` memory. The default `K = O(log²(T+1))` therefore gives **O(T log⁴(T+1)) time** and **O(T log²(T+1)) memory**. A custom region count multiplies the cost by K. Shorter queries can reuse the prepared rules; setup still covers the configured horizon. Numerical moment preprocessing is also nearly linear in T, up to logarithmic factors in T and `1/moment_tol`.

An optional `moment_provider(prior, lower, upper, degree)` must return `degree+1` normalized conditional Chebyshev moments, with zeroth moment 1. `prior` is `"co96"` (coordinate p) or `"r70"` (coordinate beta); the local coordinate is `(2*coordinate-lower-upper)/(upper-lower)`. Interval masses are evaluated from the continuous priors' analytic formulas. This interface separates the exact integration algorithm from how its input moments are obtained. NumPy still stores the returned values in floating point.

The built-in provider uses positive angular quadrature and FFT summation of its cosine moments. It resolves the central R70 singularity using symmetry and analytic tail mass, with a cutoff that shrinks with degree and tolerance. The tolerance controls the Fourier truncation and central tail approximation; remaining Gaussian quadrature error is assessed by increasing `moment_oversample`.

Computed regional allocations are projected onto their intervals. Regional weights use past wealth, and reported wealth accumulates the actual returns of the combined allocations. This retains the predictable betting strategy's null validity.

## Validation and benchmarks

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest -v test_opt_fft.py
OPENBLAS_NUM_THREADS=1 python benchmark_opt_fft.py --repeats 1
```

All 16 tests pass, covering conditional moments, independent integrals, analytic Bernoulli wealth, reversal accuracy, prefix predictability, accumulated returns, null-path probabilities, and the command-line API.
The benchmark covers 15 paths: lengths 512, 2,048, and 8,192 on Bernoulli and bounded reversals, two Bernoulli alternatives, and Beta(4,3) observations. It compares the default engine with a refined positive grid, checks Bernoulli CO96 wealth analytically, and independently integrates final reversal wealth. The maximum mixture log-wealth discrepancy against the proposed grid was 6.7e-11, with no projected regional bets. The saved timings use one evaluation per case, excluding moment/rule setup, which is recorded separately. Use `--repeats 3` for repeated timings.

Full results: [benchmark.json](results/benchmark.json). The direct evaluator in `validation.py` is a reference for tests and experiments.


## Run the experiments

```bash
OPENBLAS_NUM_THREADS=1 python demo.py --n 2048 --trials 5 --validate
```

The included plots and results use the small demonstration above (five trials per alternative, horizon 2,048); they are illustrative Monte Carlo summaries. The OPT-FFT demo simulates only Bernoulli means `mu0 + gap`, above the null. The default means are `0.7, 0.6, 0.55, 0.525`; smaller gaps are harder to detect. Customize the experiment with these options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--mu0` | `0.5` | Null mean |
| `--alpha` | `0.05` | Type-I error level |
| `--gaps` | `0.2 0.1 0.05 0.025` | Positive gaps above the null, each smaller than `1-mu0` |
| `--trials` | `100` | Independent trials per alternative |
| `--n` | `10000` | Observations per trial |
| `--seed` | `2026` | Random seed |
| `--output-dir` | `results/` | Output directory |

Add `--validate` to run a numerical check on the first 256 observations of the
first trial in each scenario.

Curves show average log wealth over all trials, including after rejection, with shading of ±1 standard error. The display ends 10% above the individual test's log rejection threshold, while saved traces cover the full horizon.

Stopping-time averages count trials that have not rejected by the horizon as having reached the horizon. They therefore summarize capped stopping times.

![OPT-FFT mean tests with decreasing gaps](results/mean_tests.png)

[Summary](results/summary.csv) · [Individual trials](results/trials.csv) ·
[Detailed results](results/mean_tests.json) · [PDF figure](results/mean_tests.pdf)

Each scenario also saves `trace_gap_<gap>_above.csv`, with columns
`n,mean_log_wealth,se_log_wealth,trials`. With one trial, shading is omitted, and
the standard-error column is zero. All outputs are in [results/](results/).
