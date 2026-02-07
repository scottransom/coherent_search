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
    r_int = int(np.floor(r)) + 1 if r == np.floor(r) else int(np.ceil(r))
    half_m = m // 2
    return ft[r_int - half_m : r_int + half_m]


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
