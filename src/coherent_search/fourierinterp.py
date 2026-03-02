import numpy as np
import numpy.typing as npt
import coherent_search.utils as utils


def get_finterp_coeffs(dr: float, m: int) -> npt.NDArray:
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
    coeffs = get_finterp_coeffs(r % 1.0, m)
    bins = get_nearby_fourier_bins(r, ft, m)
    return np.dot(coeffs.conjugate(), bins)


def get_finterp_multi_coeffs(rs: npt.NDArray, m: int) -> npt.NDArray:
    """
    Compute Fourier interpolation coeffs for multiple real-valued Fourier frequencies

    Parameters:
        rs (np.ndarray): Real-valued Fourier frequencies to interpolate (all between 2 bins).
        m (int): Number of interpolation coefficients (even).

    Returns:
        np.ndarray: Array of Fourier interpolation coefficients.
    """
    lo_rint = int(np.floor(rs.min() + 1e-15))
    hi_rint = int(np.floor(rs.max() + 1e-15))
    assert hi_rint - lo_rint == 0, "rs must all be between 2 Fourier bins"
    assert m % 2 == 0, "m must be even"
    offsets = (rs % 1.0)[:, np.newaxis] - np.arange(-m // 2 + 1, m // 2 + 1)
    coeffs = np.sinc(offsets) * np.exp(1j * np.pi * offsets)
    return coeffs


def finterp_multi(rs: npt.NDArray, ft: npt.NDArray, m: int, coeffs=None) -> npt.NDArray:
    """
    Perform Fourier interpolation at multiple real-valued Fourier frequencies

    Parameters:
        rs (np.ndarray): Real-valued Fourier frequencies to interpolate (all between 2 bins).
        ft (np.ndarray): Fourier transform array.
        m (int): Number of interpolation coefficients (even).
        coeffs (np.ndarray, optional): Precomputed Fourier interpolation coefficients for rs and m.

    Returns:
        np.ndarray: Interpolated Fourier amplitudes at frequencies rs.
    """
    if coeffs is not None:
        assert coeffs.shape == (len(rs), m), "coeffs shape must be (len(rs), m)"
    else:
        coeffs = get_finterp_multi_coeffs(rs, m)
    bins = get_nearby_fourier_bins(rs[0], ft, m)
    return np.vecdot(coeffs, bins)


def get_finterp_FFT_coeffs(numbetween: int, m: int, fftlen: int) -> npt.NDArray:
    """
    Compute Fourier interpolation coeffs for FFT correlation method

    Parameters:
        numbetween (int): The number of interpolated points between each FFT bin.
        m (int): Number of interpolation coefficients (even).
        fftlen (int): Length of the FFT to use for correlation (must be >= numbetween * m).

    Returns:
        np.ndarray: FFT'd Fourier interpolation coefficients ready for correlation.
    """
    assert m % 2 == 0, "m must be even"
    assert fftlen >= numbetween * m, "fftlen must be >= numbetween * m"
    assert fftlen == utils.next_pow_of_2(fftlen), "fftlen must be a power of 2"
    # Get and prep the interpolation coefficients
    coeffarr = np.zeros(fftlen, dtype=np.complex128)
    offsets = np.arange(numbetween * m // 2) / numbetween
    coeffarr[: len(offsets)] = np.sinc(offsets) * np.exp(-1j * np.pi * offsets)
    offsets = (-(offsets + 1.0 / numbetween))[::-1]
    coeffarr[-len(offsets) :] = np.sinc(offsets) * np.exp(-1j * np.pi * offsets)
    return np.fft.fft(coeffarr).conjugate()


def finterp_FFT(
    lobin: int, numbins: int, numbetween: int, ft: npt.NDArray, m: int, coeffs=None
) -> npt.NDArray:
    """
    Perform Fourier interpolation for many frequencies using FFT correlation

    Parameters:
        lobin (int): The integer FFT bin number for the lowest return value.
        numbins (int): The number of returned FFT bins to cover with interpolation.
        numbetween (int): The number of interpolated points between each FFT bin.
        ft (np.ndarray): Fourier transform array.
        m (int): Number of interpolation coefficients (even).
        coeffs (np.ndarray, optional): Precomputed FFT'd interpolation coefficients.

    Returns:
        np.ndarray: Interpolated Fourier amplitudes at requested frequencies.
                    (freqs = lobin + np.arange(numbins * numbetween) / numbetween)
    """

    m2 = m // 2
    numftbins = (numbins + m) * numbetween
    fftlen = utils.next_pow_of_2(numftbins)

    # Get and the interpolation coefficients if needed
    if coeffs is not None:
        assert len(coeffs) == fftlen, "coeffs length must be equal to fftlen  "
    else:
        coeffs = get_finterp_FFT_coeffs(numbetween, m, fftlen)

    # Get and prep the Fourier amplitudes
    ftarr = np.zeros(fftlen, dtype=np.complex128)
    tmplobin = lobin - m2
    tmphibin = lobin + numbins + m2
    ftarr[np.arange(numbins + m) * numbetween] = ft[tmplobin:tmphibin]

    # Perform the complex cross correlation
    corr = np.fft.ifft(np.fft.fft(ftarr) * coeffs)
    return corr[m2 * numbetween : (m2 + numbins) * numbetween]


class FourierInterpolator:
    "Class to perform running Fourier interpolation through a PRESTO FFT file"

    def __init__(
        self,
        ft: utils.fftfile,
        lobin: int,
        numbetween: int,
        m: int,
        fftlen: int,
        coeffs=None,
    ) -> None:
        """Build a Fourier interpolator that will walk through an FFT

        Parameters
        ----------
        ft : utils.fftfile
            A PRESTO FFT file object to interpolate through.
        lobin : int
            The integer FFT bin number for the lowest return value.
        numbetween : int
            The number of interpolated points between each FFT bin.
        m : int
            Number of interpolation coefficients (even).
        fftlen : int
            Length of the FFT to use for correlation (must be >= numbetween * m).
        coeffs : _type_, optional
            Precomputed Fourier interpolation coefficients for numbetween and m, by default None
        """
        self.ft = ft
        self.lobin = lobin
        self.numbetween = numbetween
        self.m = m
        self.fftlen = fftlen
        # This is the number of full FFT bins we will interpolate each time
        self.numbins = (fftlen // numbetween) - m - 1
        self.nextbin = lobin + self.numbins
        if coeffs is None:
            self.coeffs = get_finterp_FFT_coeffs(numbetween, m, fftlen)
        else:
            self.coeffs = coeffs
        self.ftamps = self.get_ftamps(lobin)

    def get_ftamps(self, lobin: int) -> np.ndarray:
        """Get the Fourier-interpolated FFT amplitudes starting at lobin

        Parameters
        ----------
        lobin : int
            The integer FFT bin number for the lowest return value.

        Returns
        -------
        np.ndarray
            The Fourier-interpolated FFT amplitudes starting at lobin.
        """
        self._rs = (
            np.arange(self.numbins * self.numbetween) / self.numbetween + self.lobin
        )
        if lobin + self.numbins + self.m // 2 >= self.ft.N // 2:
            return np.zeros_like(self._rs, dtype=np.complex128)
        else:
            return finterp_FFT(
                lobin,
                self.numbins,
                self.numbetween,
                self.ft.amps,
                self.m,
                coeffs=self.coeffs,
            )

    @property
    def rs(self) -> np.ndarray:
        """The real-valued Fourier frequencies (bins) for the current interpolation"""
        if not hasattr(self, "_rs"):
            self._rs = (
                np.arange(self.numbins * self.numbetween) / self.numbetween + self.lobin
            )
        return self._rs

    def interpolated_ftamps(self, rs: np.ndarray) -> np.ndarray:
        """Return the linear-interpolated Fourier amplitudes at the Fourier frequencies rs

        Parameters
        ----------
        rs : np.ndarray
            Fourier frequencies at which to interpolate the current FFT amplitudes.

        Returns
        -------
        np.ndarray
            The linear-interpolated FFT amplitudes at the given Fourier frequencies rs.
        """
        if rs.max() > self.rs[-1]:
            self.lobin = int(np.floor(rs.min()))
            self.ftamps = self.get_ftamps(self.lobin)
            self.nextbin = self.lobin + self.numbins
        return np.interp(rs, self.rs, self.ftamps)
