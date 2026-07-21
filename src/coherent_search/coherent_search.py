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
    """
    wmax = max(1, int(maxfrac * nbins))
    widths = []
    w = 1
    while w <= wmax:
        widths.append(w)
        w = max(int(fsp * w), w + 1)
    return widths


def snr_metric(profs, fsp=1.5, maxfrac=0.3):
    """Peak boxcar matched-filter signal-to-noise metric for each profile.

    Correlates each profile with a fixed bank of top-hat (boxcar) filters and
    returns the peak matched-filter S/N,

        metric = max_{w, phase} (sum_{i=phase}^{phase+w-1}(P_i - median)) / (sigma * sqrt(w))

    computed with the riptide prefix-sum "strided differences" (Morello et al.
    2020, MNRAS 497, 4654, Sec. 5.4): one circular prefix sum of the
    median-subtracted profile, after which every boxcar sum is a two-index
    difference.  Because the widths are chosen a priori (not from the data), a
    width-w boxcar over white noise is N(0, w*sigma**2), so dividing by sqrt(w)
    makes every (phase, width) trial exactly N(0, 1): the peak over trials follows
    analytic extreme-value statistics with a known, ~nbins-flat trials factor, and
    there is no sqrt(nbins) noise floor to normalize away (which the older on-pulse
    metrics suffered, biasing a fixed threshold toward high-bin-count profiles).

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

    # Per-profile baseline: subtract each profile's own median.
    prof0 = profs - np.median(profs, axis=1, keepdims=True)

    widths = boxcar_widths(nbins, fsp, maxfrac)
    wmax = widths[-1]
    # Circular prefix sum: tile by wmax so wrap-around boxcars read real data.
    #   csum[:, k] = sum of the first k tiled bins,  boxcar(phase, w) = csum[phase+w] - csum[phase]
    tiled = np.concatenate((prof0, prof0[:, :wmax]), axis=1)
    csum = np.zeros((nprofs, nbins + wmax + 1))
    np.cumsum(tiled, axis=1, out=csum[:, 1:])

    best = np.full(nprofs, -np.inf)
    for w in widths:
        # All nbins phases at once; peak matched-filter S/N over phase for this width.
        boxsums = csum[:, w : w + nbins] - csum[:, :nbins]
        best = np.maximum(best, boxsums.max(axis=1) / (sigma * np.sqrt(w)))
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
