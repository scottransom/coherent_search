# How many terms does brute-force Fourier interpolation actually need?

**What this is.** A numerical study of the fractional accuracy of the direct
(brute-force) Fourier interpolation in `coherent_search.fourierinterp` as a
function of the number of summed terms `m`, with realistic noise. The immediate
purpose is to set the default `m` in the sister Julia code
[`CoherentSearch.jl`](https://github.com/scottransom/CoherentSearch.jl), where
the interpolation is now the direct `O(m)` summation and its cost is therefore
**linear in `m`** — so any `m` larger than needed is time thrown away.

Experiment: [`interp_accuracy_vs_m.py`](interp_accuracy_vs_m.py). It writes the
four-panel figure `interp_accuracy_vs_m.png`, which is *not* in git
(`examples/*.png` is gitignored) — run the script to regenerate it; see §7.

**Bottom line: `m = 16` is the knee.** It halves the interpolation work relative
to the current `m = 32` default and costs 0.63% in recovered signal power, on a
sensitivity budget already dominated by a ~6.5% trial-grid loss. Details and the
supporting numbers are in §5.

---

## 1. Setup

Following the model requested: the signal sits at a Fourier frequency uniformly
distributed within a bin (`dr ~ U[0,1)`), and the surrounding bins carry white
noise whose powers are `chi^2` with 2 DOF, normalized to unit mean power (PRESTO
convention). Every bin out to ±4096 contains **both**

```
bin_k  =  A·sinc(δ_k)·e^{iπδ_k}  +  n_k,        δ_k = dr − k
          ^ the signal's sinc leakage             ^ independent complex Gaussian
```

so the signal's tails into distant bins are fully included — that leakage is the
entire reason `m` matters at all. Signal strength is quoted as the normalized
power `P = |A|²`, i.e. the power a perfect interpolation would measure. A blind
search detects single harmonics at roughly `P ~ 20–40`.

The semi-analytic bin model is cross-checked against `fourier_interp()` run on an
actual FFT of a real sinusoid (which contains the true leakage, the
negative-frequency image, and finite-`N` Dirichlet-kernel corrections by
construction). They agree to **5×10⁻⁵** over all `m` and several `dr`, and the
noise-only interpolated power reproduces the predicted `S_m` to within the
Monte Carlo error. Monte Carlo: 20 000 trials, seed 20260809.

## 2. The structure of the problem

The interpolation forms `X_m = Σ_k conj(c_k)·bin_k` with
`c_k = sinc(δ_k)·e^{iπδ_k}`. Because a sinusoid's response in bin `k` is *exactly*
that same function, each term reduces to

```
conj(c_k) · bin_k  =  A·e^{iφ}·sinc²(δ_k)  +  (noise)
```

Two consequences follow immediately, and they drive everything else:

**(a) The truncation weight is real and positive.** Define

```
S_m(dr) = Σ_k sinc²(dr − k)        over the m offsets −m/2+1 … m/2,     S_∞ = 1
```

Then the noiseless result is `X_m = A·e^{iφ}·S_m` — an amplitude scaled by a
**real** factor. So truncating the sum introduces **exactly zero phase error**,
at any `m`. This is worth stating plainly for a coherent harmonic-summing code:
small `m` cannot degrade the coherence of the harmonic sum. The residual phase
errors in panel (d) of the figure are entirely noise in the omitted bins: at
`m = 16` the median `|Δφ|` is 0.74° at a marginal `P = 10` and 0.43° at `P = 30`,
and even `m = 8` only reaches 1.04° / 0.61°. `cos(1°) = 0.99985`, so the
coherence penalty is negligible at any `m` in play.

**(b) Truncation discards signal and noise in the same proportion.** The
accumulated noise variance is also `S_m`. So the signal-to-noise ratio goes as
`A·S_m/√S_m = A·√S_m`, and the fraction of signal **power** recovered is exactly
`S_m`. One number governs the whole trade-off.

## 3. The truncation loss: 1 − S_m ≈ 0.2/m

The lost power fraction follows the large-`m` asymptote closely from `m = 2` on:

```
1 − S_m(dr)  ≈  4·sin²(π·dr) / (π²·m)      →   0.405/m worst case (dr = 0.5)
                                               0.203/m averaged over dr
```

The `sin²(π·dr)` means the loss vanishes at bin centers and peaks halfway
between. The `dr`-average happens to coincide exactly with the `dr = 0.25` case,
since `⟨sin²⟩ = ½`.

| m | mean loss | worst case (dr=0.5) | ripple over 32 harmonics | cost vs m=32 |
|---:|---:|---:|---:|---:|
| 4 | 5.01% | 9.94% | 0.62% | 0.125× |
| 6 | 3.36% | 6.69% | 0.42% | 0.188× |
| 8 | 2.53% | 5.04% | 0.32% | 0.25× |
| 12 | 1.69% | 3.37% | 0.21% | 0.375× |
| **16** | **1.27%** | **2.53%** | **0.16%** | **0.5×** |
| 24 | 0.84% | 1.69% | 0.11% | 0.75× |
| 32 *(current)* | 0.63% | 1.27% | 0.08% | 1.0× |
| 64 | 0.32% | 0.63% | 0.04% | 2.0× |

The "ripple" column matters for a coherent harmonic sum. Each harmonic `h` lands
at its own effectively-random `dr_h`, and the coherent sum's sensitivity depends
on the *average* of `S_m` over the summed harmonics. With `nharms = 32` the
`dr`-dependent scalloping averages down by `√32`, leaving the mean loss as an
essentially **deterministic** offset rather than a source of candidate-to-candidate
scatter. Use the mean-loss column, not the worst case, when budgeting.

## 4. With noise, the measured power is dominated by the noise — but that is not the whole story

This was the counter-intuitive part of the study. Writing `Y = |X_m|²/S_m` for
the recovered normalized power:

```
E[Y]/P − 1  =  1/P − (1 − S_m)            bias:  noise pedestal, and truncation
sd(Y)/P     =  √(2·S_m/P + 1/P²)  →  √(2/P)   scatter: independent of m
```

(Both verified against the Monte Carlo to 1–2%.) The scatter is 26% at `P = 30`,
14% at `P = 100`, 4.5% at `P = 1000`, and **`m` does not appear in it** — because
of §2(b), truncation shrinks signal and noise together. The truncation term
enters only as a bias, worth 1.3% at `m = 16`. Against 26% of noise scatter it
adds 0.03% in quadrature. The two are equal only at `P_equal ≈ 2/(1−S_m)² ≈ 50·m²`,
i.e. `P ~ 12 500` at `m = 16` — far beyond any real detection.

So for **measuring one candidate's power**, `m` is irrelevant past about 4–8.

**But a bias and a variance are not interchangeable for a search**, even at 20:1.
The noise scatter is random, averages out across candidates, and is already
folded into the detection threshold statistics. The `(1 − S_m)` loss is
systematic — every candidate's power is low by the same fraction, always in the
same direction, and no amount of averaging removes it. The threshold calibration
never sees it. That is why the choice of `m` should be made from §3, not from the
noisy panel (c).

> A wrinkle in panel (c) of the figure, since it is what prompted this section:
> the plotted bias mixes the `+1/P` noise pedestal (m-independent, normally
> subtracted) with the `−(1−S_m)` truncation term. Their accidental cancellation
> near `P = 30, m ≈ 6` is a coincidence of the two, not a sweet spot.

## 5. Recommendation for CoherentSearch.jl

Current state (`src/search.jl:55`): `m = 32`. (`numbetween = 16` is the separate
`:fft`-path oversampling — likely the source of the "m = 16" recollection.)

The right frame is a **sensitivity budget**, comparing the truncation loss to the
other systematic losses already accepted in the search. The dominant one is the
trial-frequency grid. For a step of `Δr` bins the mean power loss from landing
off-peak is:

| grid step Δr | mean power loss | worst case |
|---:|---:|---:|
| 0.25 | 1.7% | 5.0% |
| **0.50** *(`hidr` default)* | **6.5%** | **18.9%** |
| 1.00 | 22.6% | 59.5% |

With `hidr = 0.5` set as the step *at the highest harmonic*, the grid alone costs
~6.5% of the signal power there; lower harmonics use a finer `deltar_h` and lose
less, so 6.5% is the worst-harmonic figure rather than a flat average. It is
still the right thing to compare against, since the truncation loss applies
equally to every harmonic. Against that,
`m = 32`'s 0.63% is noise in the budget — it is buying precision that the grid
spacing has already spent. **Going to `m = 16` halves the interpolation work for
an extra 0.63% power loss**, taking the combined loss from ~7.1% to ~7.7%. In
amplitude that is 0.32%; for a Euclidean population (`N ∝ S^{−3/2}`) it is under
0.5% of detections. That is a clearly good trade.

`m = 8` is defensible but I would not make it the default: 2.53% starts to be a
real fraction of the grid loss (combined ~8.9%, ~1.4% of detections) for a
further saving that is only half as large in absolute terms. The `1/m` scaling
means the returns diminish going down just as they do going up — most of the
available time saving is already captured by the first halving.

**Suggested: default `m = 16`, and measure the resulting end-to-end speedup**
before considering `m = 8`. The interpolation's share of total runtime sets how
much this is worth; `fourier_interpolation_tradeoffs.md` puts FFTW at 12.9% of a
`:direct` search and the profile metric as the dominant cost, so the realistic
end-to-end gain is likely a few percent, not 2×.

Two things to leave alone:

- **`candidate_profile` (`src/candidate.jl:37`) uses `m = 64`.** Keep it. It runs
  a handful of times on final candidates where cost is irrelevant, and the extra
  accuracy is free there.
- **`m` must stay even**, as the coefficient range `−m/2+1 … m/2` requires.

## 6. What this study does not cover

- **Red noise / non-white spectra.** The bins here are white. Steep red noise
  changes the relative weight of the distant bins the truncation discards.
- **Nearby interfering signals** — a second pulsar, RFI, or an adjacent harmonic
  within `m/2` bins. A strong neighbour is picked up with weight `sinc²(δ)`, and
  a *larger* `m` reaches further out to collect more of it. This is the one
  mechanism that could argue for smaller `m`, and it is untested here.
- **Binary / accelerated signals**, where the response is no longer a pure sinc
  and the coefficients are not matched to the true bin content.

## 7. Reproducing

```
pixi run python examples/interp_accuracy_vs_m.py            # 20 000 trials, ~1 min
pixi run python examples/interp_accuracy_vs_m.py --ntrials 2000 --no-verify
```

Writes the table above to stdout and the four-panel figure to
`examples/interp_accuracy_vs_m.png`. `--seed` changes the realization; `MS`,
`POWERS` and `MMAX` at the top of the script set the grid of `m`, the signal
powers, and the term count standing in for `m → ∞`.
