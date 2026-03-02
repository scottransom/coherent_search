# %%
import sys
import os
import argparse
import numpy as np
import coherent_search.utils as utils
import coherent_search.fourierinterp as fi
from tqdm import tqdm


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

    # Calculate the cached Fourier interpolation coefficients for the given numbetween, nharms, and fftlen
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
    rstosearch = np.arange(numtosearch) * args.hidr / args.nharms + currentlobin
    numiters = (
        int(
            (args.hifreq * ft.T - currentlobin)
            / (numtosearch * args.hidr / args.nharms)
        )
        + 1
    )

    # Walk through the FFT file
    for _ in tqdm(range(numiters)):
        # Get the Fourier-interpolated FFT amplitudes for each harmonic
        ftprofs = np.zeros((numtosearch, args.nharms + 1), dtype=np.complex128)
        for ii in range(1, args.nharms + 1):
            ftprofs[:, ii] = fis[ii].interpolated_ftamps(rstosearch * ii)

        # Inverse FFT the interpolated Fourier amplitudes to get the pulse profiles at each trial frequency
        profs = np.fft.irfft(ftprofs, axis=1)

        # Calculate the coherent harmonic fold metric (max profile value) at each trial frequency
        maxmetric = np.max(profs, axis=1) / np.abs(np.min(profs, axis=1))

        # Pick candidates above the threshold and save them to a list
        candidates = np.where(maxmetric > args.threshold)[0]
        if len(candidates) > 0:
            for ii in candidates:
                print(
                    f"Candidate at {rstosearch[ii] / ft.T:.6f} Hz with S/N {maxmetric[ii]:.2f}"
                )

        currentlobin += numtosearch * args.hidr / args.nharms
        rstosearch = np.arange(numtosearch) * args.hidr / args.nharms + currentlobin


if __name__ == "__main__":
    main_cli()
