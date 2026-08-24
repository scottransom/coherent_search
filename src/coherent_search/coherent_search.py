# %%
import sys
import argparse
import numpy as np
import coherent_search.utils as utils
import coherent_search.fourierinterp as fi
from tqdm import tqdm


def boxcar_widths(nbins, fsp=1.5, maxfrac=0.3):
    """Geometric bank of boxcar widths (in profile bins) for an nbins-bin profile.

    w0 = 1, w_{k+1} = max(floor(fsp * w_k), w_k + 1), truncated at
    floor(maxfrac * nbins) (the widest duty cycle worth testing).  The fsp=1.5
    recurrence is riptide's default (Morello et al. 2020) and reproduces the
    hand-picked [1, 2, 3, 4, 6, 9, 13, 19, ...] sequence.  Always contains at
    least the width-1 (single-bin) filter.

    w is additionally capped at nbins - 1, as riptide's check_trial_widths also
    requires: the zero-mean unit-L2 template of a full-width boxcar is the zero
    vector.  At the default maxfrac = 0.3 the cap only bites for nbins <= 3.
    """
    if nbins < 2:
        raise ValueError(f"boxcar_widths needs nbins >= 2, got {nbins}")
    wmax = min(max(1, int(maxfrac * nbins)), int(nbins) - 1)
    widths = []
    w = 1
    while w <= wmax:
        widths.append(w)
        w = max(int(fsp * w), w + 1)
    return widths


def snr_metric(profs, fsp=1.5, maxfrac=0.3):
    """Peak boxcar matched-filter signal-to-noise metric for each profile.

    Correlates each profile with a fixed bank of top-hat (boxcar) filters and
    returns the peak matched-filter S/N.  This is riptide's statistic exactly
    (Morello et al. 2020, MNRAS 497, 4654, Sec. 5.4; ``cpp/snr.hpp:snr1``): the
    width-w template is the boxcar made **zero-mean and unit-L2**,

        height    h = sqrt((n - w) / (n * w))   on the w on-pulse bins
        baseline -b,  b = w / (n - w) * h       on the other n - w bins

    so that sum(t) = 0 and sum(t**2) = 1, and the correlation <t, P> / sigma has
    variance exactly 1 per (phase, width) under white noise.  Since b = d * (h+b)
    with duty cycle d = w / n, that correlation is

        metric = max_{w, phase} (boxsum(phase, w) - d * P.sum()) /
                                (sigma * sqrt(w * (1 - d)))

    computed with the riptide prefix-sum "strided differences": one circular
    prefix sum per profile, after which every boxcar sum is a two-index
    difference.  Written this way the statistic is manifestly invariant to any
    constant already removed from P, so the mean subtraction below is for
    numerical conditioning only.

    Because every (phase, width) trial is exactly N(0, 1), the peak over trials
    follows analytic extreme-value statistics with a known, ~nbins-flat trials
    factor: a fixed threshold means the same false-alarm rate at every width and
    every profile length, with no sqrt(nbins) floor to normalize away (which the
    older on-pulse metrics suffered, biasing a fixed threshold toward
    high-bin-count profiles).

    **Changed 2026-08-24.**  This used to subtract each profile's own *median*
    and divide by sigma * sqrt(w), which is the above times a factor that is not
    constant: sqrt(1 - d) under pure noise (median ~ 0), rising towards
    1 / sqrt(1 - d) for a bright pulse (median tracks the off-pulse level).  The
    median recovers nothing the boxcar had not already seen -- with the profile
    mean pinned at 0 the off-pulse sum is identically -boxsum, so subtracting the
    true off-pulse level is exactly a 1 / (1 - d) rescale -- while costing
    detection power at every duty cycle above a few percent, and making the
    metric's normalization depend on source brightness.  See
    ``CoherentSearch.jl`` ``Summary_and_Future_Work.md`` Sec. 3.2.

    The per-bin noise `sigma` is estimated once for the whole block (1.4826 * MAD
    pooled over all profiles), not per profile: a per-profile MAD (only nbins
    samples) has ~0.76/sqrt(nbins) relative error that would multiply straight into
    every S/N and re-inflate the small-nbins tail.  Pooling thousands of block bins
    drops its variance below a percent, and being median-based it is immune to the
    rare signal/RFI bin.  The metric is scale-free (a ratio of two
    linear-in-amplitude quantities), so it does not care whether `profs` came from a
    normalized or unnormalized inverse FFT.

    Parameters
    ----------
    profs : np.ndarray
        A 2D array of (nprofs, nbins) pulse profiles.
    fsp : float
        Geometric width-recurrence factor for the boxcar bank (default 1.5).
    maxfrac : float
        Widest boxcar as a fraction of nbins (default 0.3).

    Returns
    -------
    np.ndarray
        The peak boxcar S/N for each of the nprofs profiles (0.0 for a degenerate,
        flat block).
    """
    nprofs, nbins = profs.shape
    # One robust per-bin noise sigma for the whole block (pooled MAD).
    med = np.median(profs)
    sigma = 1.4826 * np.median(np.abs(profs - med))
    if sigma <= 0:
        return np.zeros(nprofs)

    # Conditioning only -- the statistic below is invariant to this (see above).
    prof0 = profs - profs.mean(axis=1, keepdims=True)

    widths = boxcar_widths(nbins, fsp, maxfrac)
    wmax = widths[-1]
    # Circular prefix sum: tile by wmax so wrap-around boxcars read real data.
    #   csum[:, k] = sum of the first k tiled bins,  boxcar(phase, w) = csum[phase+w] - csum[phase]
    tiled = np.concatenate((prof0, prof0[:, :wmax]), axis=1)
    csum = np.zeros((nprofs, nbins + wmax + 1))
    np.cumsum(tiled, axis=1, out=csum[:, 1:])
    stot = csum[:, nbins] - csum[:, 0]          # profile total, baseline removed

    best = np.full(nprofs, -np.inf)
    for w in widths:
        # All nbins phases at once; peak matched-filter S/N over phase for this width.
        duty = w / nbins
        boxsums = csum[:, w : w + nbins] - csum[:, :nbins]
        cand = (boxsums.max(axis=1) - duty * stot) / (sigma * np.sqrt(w * (1.0 - duty)))
        best = np.maximum(best, cand)
    return best


def main_cli():
    parser = argparse.ArgumentParser(
        description="Search a PRESTO-style FFT file for pulsations using coherent harmonic folding.",
        epilog="""In general, the FFT file should probably be barycentered, have known
RFI zapped, and have rednoise removed. Barycentering happens by default if you 
use `prepdata` or `prepsubband`. Zapping can be done using, for instance, 
`simple_zapbirds.py`, and rednoise can be removed using `rednoise` on the FFT file.
The sigma threshold is single-trial and based on equivalent gaussian sigma.
If no output candidate file name is given, the results will be written to stdout.
""",
    )
    parser.add_argument("fftfile", nargs="*", help="PRESTO FFT file to be searched.")
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=8,
        help="S/N cutoff for picking candidates (default=8)",
    )
    parser.add_argument(
        "-o", "--outputfilenm", type=str, help="Output filename to record candidates"
    )
    parser.add_argument(
        "-n",
        "--nharms",
        type=int,
        default=32,
        help="Number of harmonics to sum. A power-of-two. (default=32)",
    )
    parser.add_argument(
        "--ncands",
        type=int,
        default=100,
        help="Maximum number of candidates to return (default=100)",
    )
    parser.add_argument(
        "--lobin",
        type=int,
        default=100,
        help="Lowest frequency bin to search (default=100)",
    )
    parser.add_argument(
        "--lofreq",
        type=float,
        default=0.1,
        help="Lowest frequency (in Hz) to search (default=0.1)",
    )
    parser.add_argument(
        "--hifreq",
        type=float,
        default=100.0,
        help="Highest frequency (in Hz) to search (default=100.0)",
    )
    parser.add_argument(
        "--hidr",
        type=float,
        default=0.5,
        help="Fourier bin resolution at highest harmonic (default=0.5)",
    )
    parser.add_argument(
        "--numbetween",
        type=int,
        default=16,
        help="Number of points to interpolate between Fourier bins (default=16)",
    )
    parser.add_argument(
        "--fftlen",
        type=int,
        default=16384,
        help="Number bins in FFTs for Fourier interpolation (default=16384)",
    )
    parser.add_argument(
        "--noremove",
        action="store_true",
        help="Do not filter duplicate or harmonically-related candidates",
    )
    args = parser.parse_args()
    if not args.fftfile:
        parser.print_help()
        sys.exit(1)

    m = 32  # Number of Fourier bins to use for interpolation kernel

    # Open the PRESTO FFT file
    ft = utils.fftfile(args.fftfile[0])

    # Calculate the cached Fourier interpolation coefficients for the
    # given numbetween, nharms, and fftlen
    coeffs = fi.get_finterp_FFT_coeffs(args.numbetween, m, args.fftlen)

    # Prep the FouierInterpolator class instances
    fis = {}
    for ii in range(1, args.nharms + 1):
        fis[ii] = fi.FourierInterpolator(
            ft, args.lobin * ii, args.numbetween, m, args.fftlen, coeffs
        )

    # Number of bins to search each iteration
    numtosearch = 1024
    # Prioritize --lofreq if given, otherwise use --lobin
    currentlobin = args.lofreq * ft.T
    if args.lobin != 100:
        currentlobin = args.lobin
    # Fourier freq step-size at the fundamental frequency
    lodr = args.hidr / args.nharms
    rstosearch = np.arange(numtosearch) * lodr + currentlobin
    numiters = int((args.hifreq * ft.T - currentlobin) / (numtosearch * lodr)) + 1

    # Walk through the FFT file
    for _ in tqdm(range(numiters)):
        # Get the Fourier-interpolated FFT amplitudes for each harmonic
        ftprofs = np.zeros((numtosearch, args.nharms + 1), dtype=np.complex128)
        for ii in range(1, args.nharms + 1):
            ftprofs[:, ii] = fis[ii].interpolated_ftamps(rstosearch * ii)

        # Inverse FFT the interpolated Fourier amplitudes to get the
        # pulse profiles at each trial frequency

        # TODO: need to check if the transpose of this is faster
        profs = np.fft.irfft(ftprofs, axis=1)

        # Calculate the peak boxcar matched-filter S/N of the coherent harmonic
        # fold at each trial frequency.
        metric = snr_metric(profs)

        # Pick candidates above the threshold and save them to a list
        candidates = np.where(metric > args.threshold)[0]
        if len(candidates) > 0:
            for ii in candidates:
                print(
                    f"Candidate at {rstosearch[ii] / ft.T:.6f} Hz with S/N {metric[ii]:.2f}"
                )

        currentlobin += numtosearch * lodr
        rstosearch = np.arange(numtosearch) * lodr + currentlobin


if __name__ == "__main__":
    main_cli()
