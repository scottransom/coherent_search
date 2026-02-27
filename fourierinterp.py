import numpy as np
import numpy.typing as npt


def get_fourier_interp_coeffs(dr: float, m: int) -> npt.NDArray:
    """
    Compute m Fourier interpolation coeffs for Fourier frequency offset dr

    This routine is based on Eqn 30 from https://arxiv.org/pdf/astro-ph/0204349

    Parameters:
        dr (float): Fourier frequency offset in bins [0-1).
        m (int): Number of coefficients to compute (even).

    Returns:
        np.ndarray: Array of Fourier interpolation coefficients.
    """
    assert m % 2 == 0, "m must be even"
    assert 0.0 <= dr < 1.0, "dr must be in [0.0, 1.0)"
    offsets = dr - np.arange(-m // 2 + 1, m // 2 + 1)
    coeffs = np.sinc(offsets) * np.exp(1j * np.pi * offsets)
    return coeffs


def get_nearby_fourier_bins(r: float, ft: npt.NDArray, m: int) -> npt.NDArray:
    """
    Get the Fourier bins around a real-valued Fourier frequency r

    Parameters:
        r (float): Real-valued Fourier frequency.
        ft (np.ndarray): Fourier transform array.
        m (int): Number of bins to return (even).

    Returns:
        np.ndarray: Array of complex Fourier amplitudes.
    """
    assert m % 2 == 0, "m must be even"
    r_int = int(np.floor(r + 1e-15)) + 1
    return ft[r_int - m // 2 : r_int + m // 2]


def fourier_interp(r: float, ft: npt.NDArray, m: int) -> complex:
    """
    Perform Fourier interpolation at real-valued Fourier frequency r

    Parameters:
        r (float): Real-valued Fourier frequency.
        ft (np.ndarray): Fourier transform array.
        m (int): Number of interpolation coefficients (even).

    Returns:
        complex: Interpolated Fourier amplitude at frequency r.
    """
    assert r >= 0.0, "r must be non-negative"
    assert m % 2 == 0, "m must be even"
    coeffs = get_fourier_interp_coeffs(r % 1.0, m)
    bins = get_nearby_fourier_bins(r, ft, m)
    return np.dot(coeffs.conjugate(), bins)


def fourier_interp_multi(rs: npt.NDArray, ft: npt.NDArray, m: int) -> npt.NDArray:
    """
    Perform Fourier interpolation at multiple real-valued Fourier frequencies

    Parameters:
        rs (np.ndarray): Real-valued Fourier frequencies to interpolate (all between 2 bins).
        ft (np.ndarray): Fourier transform array.
        m (int): Number of interpolation coefficients (even).

    Returns:
        np.ndarray: Interpolated Fourier amplitudes at frequencies rs.
    """
    lo_rint = int(np.floor(rs.min() + 1e-15))
    hi_rint = int(np.floor(rs.max() + 1e-15))
    assert hi_rint - lo_rint == 0, "rs must all be between 2 Fourier bins"
    assert m % 2 == 0, "m must be even"
    offsets = (rs % 1.0)[:, np.newaxis] - np.arange(-m // 2 + 1, m // 2 + 1)
    coeffs = np.sinc(offsets) * np.exp(1j * np.pi * offsets)
    bins = get_nearby_fourier_bins(rs[0], ft, m)
    return np.vecdot(coeffs, bins)


def FFT_fourier_interp(
    lobin: int, numbins: int, numbetween: int, ft: npt.NDArray, m: int
) -> npt.NDArray:
    """
    Perform Fourier interpolation for many frequencies using FFT correlation

    Parameters:
        lobin (int): The integer FFT bin number for the lowest return value.
        numbins (int): The number of returned FFT bins to cover with interpolation.
        numbetween (int): The number of interpolated points between each FFT bin.
        ft (np.ndarray): Fourier transform array.
        m (int): Number of interpolation coefficients (even).

    Returns:
        np.ndarray: Interpolated Fourier amplitudes at requested frequencies.
                    (freqs = lobin + np.arange(numbins * numbetween) / numbetween)
    """

    m2 = m // 2
    # Get and prep the Fourier amplitudes
    numftbins = (numbins + m) * numbetween
    # The FFT length will be the next power-of-two bigger than numftbins
    ftlen = 2 ** int(np.ceil(np.log2(numftbins)))
    ftarr = np.zeros(ftlen, dtype=np.complex128)
    tmplobin = lobin - m2
    tmphibin = lobin + numbins + m2
    ftarr[np.arange(numbins + m) * numbetween] = ft[tmplobin:tmphibin]

    # Get and prep the interpolation coefficients
    coeffarr = np.zeros(ftlen, dtype=np.complex128)
    offsets = np.arange(numbetween * m2) / numbetween
    coeffarr[: len(offsets)] = np.sinc(offsets) * np.exp(-1j * np.pi * offsets)
    offsets = (-(offsets + 1.0 / numbetween))[::-1]
    coeffarr[-len(offsets) :] = np.sinc(offsets) * np.exp(-1j * np.pi * offsets)

    # Perform the complex cross correlation
    corr = np.fft.ifft(np.fft.fft(ftarr) * np.fft.fft(coeffarr).conjugate())
    return corr[m2 * numbetween : (m2 + numbins) * numbetween]
