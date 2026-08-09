#!/usr/bin/env python
"""Fractional accuracy of brute-force Fourier interpolation vs the number of terms m.

The interpolation implemented in `coherent_search.fourierinterp` estimates the
complex Fourier amplitude at a real-valued frequency r by summing m neighboring
Fourier bins weighted by the (conjugated) response function

    c_k = sinc(delta_k) * exp(i*pi*delta_k),    delta_k = dr - k

where dr = r % 1 and k runs over the bin offsets -m/2+1 ... m/2.  Since a pure
sinusoid at frequency r produces exactly that response in bin k (in the N -> inf
limit), every one of the m terms adds *coherently*, and

    X_m = sum_k conj(c_k) * b_k  ->  A * S_m(dr) + noise,
    S_m(dr) = sum_k sinc^2(dr - k)   (S_inf = 1 exactly).

Truncating at m terms therefore costs signal amplitude (a factor S_m) but also
throws away noise (the accumulated noise variance is also S_m), so the recovered
*normalized* power |X_m|^2 / S_m is an unbiased-noise estimator whose signal
content is A^2 * S_m.  The single number that matters is thus the fractional
loss 1 - S_m(dr), and how it compares to the noise-driven scatter at realistic
signal strengths.

This script measures all of that numerically:

  * The surrounding Fourier bins carry white noise: complex Gaussian amplitudes
    normalized so |b|^2 is chi^2 with 2 DOF and mean power 1 (PRESTO powers).
  * The signal sits at a frequency uniformly distributed within a Fourier bin
    (dr ~ U[0,1)), with normalized power P = |A|^2 (so P is the power you would
    measure with a perfect interpolation).
  * A verification step checks the semi-analytic bin model against the actual
    `fourierinterp.fourier_interp()` code run on a real FFT.

Usage:  pixi run python examples/interp_accuracy_vs_m.py [--ntrials N] [--seed S]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import coherent_search.fourierinterp as fi

# Colorblind-safe categorical order (Okabe-Ito), assigned in fixed order and
# backed up by distinct markers so identity never rests on color alone.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]
MARKERS = ["o", "s", "^", "D", "v"]
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d9d9d9"

# m values to test (all even, as required by the interpolation routines)
MS = np.array([2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128])
# Normalized signal powers to test.  A single-harmonic detection in a blind
# search is typically P ~ 20-40, so this brackets the interesting regime.
POWERS = [10.0, 30.0, 100.0, 1000.0]
# Number of terms used to stand in for the m -> inf "exact" interpolation
MMAX = 8192


# --------------------------------------------------------------------------
# The bin model
# --------------------------------------------------------------------------
def nesting_order(mmax: int) -> np.ndarray:
    """Bin offsets ordered so the first m of them are exactly the m used bins.

    `fourier_interp` uses offsets -m/2+1 ... m/2, so successive m share bins:
    the order is 0, 1, -1, 2, -2, 3, ...  A cumulative sum along this order
    therefore yields every X_m at once.
    """
    kk = np.empty(mmax, dtype=int)
    kk[0::2] = -np.arange(mmax // 2)
    kk[1::2] = np.arange(1, mmax // 2 + 1)
    return kk


def coherent_fraction(drs: np.ndarray, ms: np.ndarray) -> np.ndarray:
    """S_m(dr): fraction of the signal power recovered, shape (len(drs), len(ms))."""
    kk = nesting_order(int(ms.max()))
    s = np.cumsum(np.sinc(drs[:, np.newaxis] - kk) ** 2, axis=1)
    return s[:, ms - 1]


# --------------------------------------------------------------------------
# Verification of the bin model against the real fourierinterp.py code
# --------------------------------------------------------------------------
def verify_model(rng: np.random.Generator) -> None:
    """Check the analytic bin model against fourierinterp on an actual FFT."""
    print("Verifying the bin model against fourierinterp.fourier_interp()...")
    nn = 1 << 18
    tt = np.arange(nn)

    # (1) Noiseless sinusoid: |X_m| / A must equal S_m(dr).
    worst = 0.0
    for dr in (0.0, 0.13, 0.37, 0.5, 0.86):
        rr = 1000.0 + dr
        amp = 2.0 / np.sqrt(nn)  # gives normalized signal power A^2 = 1
        ts = amp * np.cos(2.0 * np.pi * rr * tt / nn + 0.7)
        ft = np.fft.rfft(ts) / np.sqrt(nn)  # unit mean noise power convention
        smeas = np.array([np.abs(fi.fourier_interp(rr, ft, m)) for m in MS])
        spred = coherent_fraction(np.array([dr]), MS)[0]
        worst = max(worst, np.abs(smeas / spred - 1.0).max())
    print(f"  noiseless signal: max |measured S_m / predicted S_m - 1| = {worst:.2e}")

    # (2) Noise only: the interpolated power must average to S_m (chi^2, 2 DOF).
    ntr = 4000
    ts = rng.standard_normal(nn)
    ft = np.fft.rfft(ts) / np.sqrt(nn)
    rs = 1000.0 + MMAX + rng.uniform(0.0, 1.0, ntr) + np.arange(ntr) * 3.0
    pw = np.array([[np.abs(fi.fourier_interp(r, ft, m)) ** 2 for m in MS] for r in rs])
    ratio = pw.mean(axis=0) / coherent_fraction(rs % 1.0, MS).mean(axis=0)
    print(
        f"  noise only:       <|X_m|^2> / S_m = {ratio.min():.3f} - {ratio.max():.3f}"
        f"  (expect 1.00 +- {np.sqrt(1.0 / ntr) * 2:.3f})"
    )
    print()


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------
def monte_carlo(
    ntrials: int, rng: np.random.Generator, chunk: int = 256
) -> dict[str, np.ndarray]:
    """Simulate signal-plus-noise interpolation for every m and signal power.

    The signal and noise parts of the sum are accumulated separately, so a
    single set of noise realizations serves every signal power (common random
    numbers), and the noiseless limit falls out for free.

    Returns per-trial metrics of shape (ntrials, len(POWERS), len(MS)).
    """
    kk = nesting_order(MMAX)
    idx = MS - 1  # cumsum index holding the sum of the first m terms
    powers = np.array(POWERS)

    out = {k: [] for k in ("dr", "s_m", "s_exact", "amp_err", "pow_err", "phs_err")}
    done = 0
    while done < ntrials:
        nb = min(chunk, ntrials - done)
        done += nb
        # Signal is uniformly distributed in Fourier frequency; its phase is
        # irrelevant (circular noise), so take it real and positive.
        drs = rng.uniform(0.0, 1.0, nb)[:, np.newaxis]
        delta = drs - kk
        sinc_d = np.sinc(delta)

        # conj(c_k) * b_k for a unit-amplitude signal, and for the noise alone.
        # noise bins: Re, Im ~ N(0, 1/2) so that |b|^2 ~ chi^2_2 with mean 1.
        noise = rng.standard_normal((nb, MMAX, 2)).view(np.complex128)[..., 0]
        noise *= np.sqrt(0.5)
        sig_cs = np.cumsum(sinc_d**2, axis=1)  # -> A * S_m
        noi_cs = np.cumsum(sinc_d * np.exp(-1j * np.pi * delta) * noise, axis=1)

        s_m = sig_cs[:, idx]  # (nb, nm)
        s_ex = sig_cs[:, -1:]  # (nb, 1)
        n_m = noi_cs[:, idx]
        n_ex = noi_cs[:, -1:]

        # broadcast over signal power: (nb, npow, nm)
        amps = np.sqrt(powers)[np.newaxis, :, np.newaxis]
        xm = amps * s_m[:, np.newaxis, :] + n_m[:, np.newaxis, :]
        xex = amps * s_ex[:, np.newaxis, :] + n_ex[:, np.newaxis, :]

        out["dr"].append(drs[:, 0])
        out["s_m"].append(s_m)
        out["s_exact"].append(np.broadcast_to(s_ex, s_m.shape).copy())
        out["amp_err"].append(np.abs(xm - xex) / np.abs(xex))
        # Normalizing by S_m keeps the noise at unit mean power, so this is the
        # fractional error of the power you would actually quote.
        out["pow_err"].append(
            (np.abs(xm) ** 2 / s_m[:, np.newaxis, :])
            / powers[np.newaxis, :, np.newaxis]
            - 1.0
        )
        out["phs_err"].append(np.degrees(np.abs(np.angle(xm * xex.conjugate()))))

    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def style_axes(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(MS[::2])
    ax.set_xticklabels([str(m) for m in MS[::2]])
    ax.minorticks_off()
    ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.set_xlabel(xlabel, color=INK, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK, fontsize=9.5)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)


def make_plots(mc: dict[str, np.ndarray], outfile: Path) -> None:
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["axes.facecolor"] = "white"
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))
    fig.patch.set_facecolor("white")
    mfloat = MS.astype(float)

    # --- (a) deterministic signal-power loss, no noise -------------------
    ax = axes[0, 0]
    drs = np.linspace(0.0, 1.0, 2001, endpoint=False)
    loss = 1.0 - coherent_fraction(drs, MS)
    for lbl, curve, color, mk in (
        ("worst case (dr = 0.5)", loss[np.argmin(np.abs(drs - 0.5))], PALETTE[1], "s"),
        # note: the dr-average equals the dr = 0.25 case exactly, since the
        # dr dependence is sin^2(pi*dr) and <sin^2> = 1/2
        ("mean over dr  (= dr = 0.25)", loss.mean(axis=0), PALETTE[0], "o"),
        ("dr = 0.1", loss[np.argmin(np.abs(drs - 0.1))], PALETTE[2], "^"),
    ):
        ax.plot(
            mfloat, curve, mk + "-", color=color, lw=1.6, ms=4.5, label=lbl, zorder=3
        )
    ax.plot(
        mfloat,
        4.0 / (np.pi**2 * mfloat),
        ":",
        color=MUTED,
        lw=1.4,
        label=r"$4/\pi^2 m$  and  $2/\pi^2 m$",
        zorder=2,
    )
    ax.plot(mfloat, 2.0 / (np.pi**2 * mfloat), ":", color=MUTED, lw=1.4, zorder=2)
    style_axes(
        ax,
        "m (number of terms in the sum)",
        r"lost signal power fraction  $1 - S_m$",
        "(a) Truncation loss alone (noise-free)",
    )
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)

    # --- (b) fractional amplitude error vs the exact interpolation --------
    ax = axes[0, 1]
    det = np.median(1.0 - mc["s_m"] / mc["s_exact"], axis=0)
    ax.plot(
        mfloat,
        det,
        "--",
        color=INK,
        lw=1.8,
        zorder=4,
        label="noiseless (truncation only)",
    )
    for i, pw in enumerate(POWERS):
        med = np.median(mc["amp_err"][:, i, :], axis=0)
        ax.plot(
            mfloat,
            med,
            MARKERS[i] + "-",
            color=PALETTE[i],
            lw=1.6,
            ms=4.5,
            label=f"P = {pw:g}",
            zorder=3,
        )
    style_axes(
        ax,
        "m (number of terms in the sum)",
        r"median  $|X_m - X_\infty| / |X_\infty|$",
        "(b) Fractional amplitude error vs. exact interpolation",
    )
    ax.legend(
        frameon=False,
        fontsize=8.5,
        labelcolor=INK,
        title="normalized signal power",
        title_fontsize=8.5,
        alignment="left",
    )

    # --- (c) fractional error of the recovered normalized power -----------
    ax = axes[1, 0]
    ax.plot(
        mfloat,
        np.median(1.0 - mc["s_m"], axis=0),
        "--",
        color=INK,
        lw=1.8,
        zorder=4,
        label="noiseless (truncation bias)",
    )
    for i, pw in enumerate(POWERS):
        rms = np.sqrt(np.mean(mc["pow_err"][:, i, :] ** 2, axis=0))
        ax.plot(
            mfloat,
            rms,
            MARKERS[i] + "-",
            color=PALETTE[i],
            lw=1.6,
            ms=4.5,
            label=f"P = {pw:g}",
            zorder=3,
        )
    style_axes(
        ax,
        "m (number of terms in the sum)",
        r"RMS fractional error in $|X_m|^2/S_m$",
        r"(c) Error in the recovered power ($S_m$-normalized)",
    )
    ax.legend(
        frameon=False,
        fontsize=8.5,
        labelcolor=INK,
        title="normalized signal power",
        title_fontsize=8.5,
        alignment="left",
    )

    # --- (d) phase error --------------------------------------------------
    ax = axes[1, 1]
    for i, pw in enumerate(POWERS):
        med = np.median(mc["phs_err"][:, i, :], axis=0)
        ax.plot(
            mfloat,
            med,
            MARKERS[i] + "-",
            color=PALETTE[i],
            lw=1.6,
            ms=4.5,
            label=f"P = {pw:g}",
            zorder=3,
        )
    style_axes(
        ax,
        "m (number of terms in the sum)",
        r"median $|\Delta\phi|$ vs. $X_\infty$ (degrees)",
        "(d) Phase error vs. exact interpolation",
    )
    ax.legend(
        frameon=False,
        fontsize=8.5,
        labelcolor=INK,
        title="normalized signal power",
        title_fontsize=8.5,
        alignment="left",
    )

    fig.suptitle(
        "Brute-force Fourier interpolation: fractional accuracy vs. number of terms m",
        color=INK,
        fontsize=12.5,
        x=0.5,
        y=0.985,
    )
    fig.text(
        0.5,
        0.955,
        "signal uniform in Fourier frequency; surrounding bins $\\chi^2$ "
        "(2 DOF) with unit mean power",
        color=MUTED,
        fontsize=9,
        ha="center",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(outfile, dpi=150, facecolor="white")
    print(f"Wrote {outfile}")


def print_table(mc: dict[str, np.ndarray]) -> None:
    drs = np.linspace(0.0, 1.0, 20001, endpoint=False)
    loss = 1.0 - coherent_fraction(drs, MS)
    mean_loss, worst_loss = loss.mean(axis=0), loss.max(axis=0)
    print("Truncation loss and RMS fractional error of the recovered power")
    print("  P_equal = signal power at which the truncation bias 1-S_m first")
    print("  matches the noise scatter sqrt(2/P) of the measured power\n")
    print(
        f"{'m':>5} {'<1-S_m>':>10} {'worst':>10} {'P_equal':>10}   "
        + "  ".join(f"{'P=' + f'{p:g}':>9}" for p in POWERS)
    )
    print(
        f"{'':>5} {'':>10} {'':>10} {'':>10}   "
        + "  ".join(f"{'RMS frac':>9}" for _ in POWERS)
    )
    print("-" * (40 + 11 * len(POWERS)))
    for j, m in enumerate(MS):
        # |X|^2 = A^2 + 2 A Re(n) + |n|^2 with Var(Re n) = 1/2, so the measured
        # power has fractional RMS -> sqrt(2/P) for P >> 1.
        p_equal = 2.0 / mean_loss[j] ** 2
        rms = [
            np.sqrt(np.mean(mc["pow_err"][:, i, j] ** 2)) for i in range(len(POWERS))
        ]
        print(
            f"{m:>5d} {mean_loss[j]:>10.2e} {worst_loss[j]:>10.2e} {p_equal:>10.3g}   "
            + "  ".join(f"{r:>9.4f}" for r in rms)
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ntrials", type=int, default=20000, help="Monte Carlo trials")
    ap.add_argument("--seed", type=int, default=20260809, help="RNG seed")
    ap.add_argument("--no-verify", action="store_true", help="skip the FFT cross-check")
    ap.add_argument(
        "--outfile",
        type=Path,
        default=Path(__file__).with_name("interp_accuracy_vs_m.png"),
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if not args.no_verify:
        verify_model(rng)

    print(f"Running {args.ntrials} trials with up to m = {MMAX} terms...")
    mc = monte_carlo(args.ntrials, rng)
    print_table(mc)
    make_plots(mc, args.outfile)


if __name__ == "__main__":
    main()
