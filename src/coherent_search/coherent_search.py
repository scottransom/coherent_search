# %%
import sys
import os
import argparse
import numpy as np
import coherent_search.utils as utils
import coherent_search.fourierinterp as fi
from tqdm import tqdm


def snr_metric(profs, ngoodbins, xsignal=0.2, metric="non", p=0.5):
    """Compute a signal-to-noise metric that is pulse-width sensitive

    This is based on the sigma definition as in single_pulse_search.py:
       metric = sum_on(signal - bkgd_level) / RMS / width**p
    The off-pulse level is the median of the profile.  The "signal" is the summed
    excess over that median for every *on-pulse* bin -- the bins that rise above
    `xsignal` of the peak-over-median height.  Summing only this on-pulse set
    (rather than the whole zero-mean profile) keeps the signal a stable measure of
    pulsed flux that does not grow with nbins, while still capturing multi-component
    pulses (e.g. two narrow peaks with a valley between them) that a boxcar sum
    would miss.  The RMS of the profile noise is expected to be ~1/sqrt(2*ngoodbins+1).

    The `width` penalty is taken over that same on-pulse set:
      metric="non": width = N_on, the count of on-pulse bins (a duty-cycle penalty).
        p=0.5 is the calibrated matched-filter normalization; larger p penalizes
        high-duty-cycle signals (broad / many-toothed RFI) more, while leaving
        narrow pulses (even separated multi-component or interpulse) alone.
      metric="sd2": width = sum(d**2), the summed squared modular phase distance of
        the on-pulse bins from the peak.  Larger p penalizes phase spread harder,
        but also down-weights genuinely separated components.

    Parameters
    ----------
    profs : np.ndarray
        A 2D array of (nprofs, nbins) pulse profiles.
    ngoodbins : int
        The number of good harmonics in the profiles (<= nbins/2+1)
    xsignal : float
        Fraction of the peak-over-median height above which a bin is "on-pulse".
    metric : str
        "non" (N_on, duty cycle) or "sd2" (sum d**2, phase spread).
    p : float
        Exponent on the width penalty (default 0.5).
    """
    nprofs, nbins = profs.shape
    bins = np.arange(nbins)
    meds = np.median(profs, axis=1)
    profs -= meds[:, np.newaxis]
    mxinds = np.argmax(profs, axis=1)
    mxs = profs[np.arange(nprofs)[:, np.newaxis], mxinds[:, np.newaxis]]
    rms = 1.0 / np.sqrt(2 * ngoodbins + 1)
    onmask = profs > xsignal * mxs
    signal = np.where(onmask, profs, 0.0).sum(axis=1)
    if metric == "non":
        width = np.maximum(onmask.sum(axis=1), 1.0)
    elif metric == "sd2":
        width = np.ones(nprofs)
        for ii in range(nprofs):  # Would like to get rid of this loop
            hibins = bins[onmask[ii]]
            bindiffs = mxinds[ii] - hibins
            bindiffs[bindiffs > nbins // 2] -= nbins
            bindiffs[bindiffs < -nbins // 2] += nbins
            width[ii] = max(np.sum(bindiffs**2), 1)
    else:
        raise ValueError("metric must be 'non' or 'sd2'")
    return signal / rms / width**p


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

        # Calculate the coherent harmonic fold metric (max profile value)
        # at each trial frequency
        # metric = np.max(profs, axis=1) / np.abs(np.min(profs, axis=1))
        metric = snr_metric(profs, min(ft.N / 2 / rstosearch.mean(), args.nharms))

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
