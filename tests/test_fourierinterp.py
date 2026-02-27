import numpy as np

from coherent_search.fourierinterp import (
    get_fourier_interp_coeffs,
    get_nearby_fourier_bins,
    fourier_interp,
    fourier_interp_multi,
    fourier_interp_FFT,
)


def test_get_fourier_interp_coeffs():
    # Test with m=10 and dr=0.1
    m = 10
    dr = 0.1
    coeffs = get_fourier_interp_coeffs(dr, m)

    # Can calculate expected values using gen_r_response() from PRESTO:
    # import presto.presto as pp
    # expected_coeffs = pp.gen_r_response(dr, 1, m+2)[2:]

    expected_coeffs = np.array(
        [
            0.02281681 + 0.00741363j,
            0.03017707 + 0.00980513j,
            0.04454711 + 0.01447423j,
            0.08504448 + 0.02763263j,
            0.9354893 + 0.3039589j,
            -0.10394325 - 0.03377321j,
            -0.04923628 - 0.01599784j,
            -0.03225825 - 0.01048134j,
            -0.0239869 - 0.00779382j,
            -0.01909162 - 0.00620324j,
        ],
        dtype=np.complex64,
    )

    np.testing.assert_allclose(coeffs.real, expected_coeffs.real, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(coeffs.imag, expected_coeffs.imag, rtol=1e-5, atol=1e-7)

    # Test with m=6 and dr=0.0
    m = 6
    dr = 0.0
    coeffs = get_fourier_interp_coeffs(dr, m)

    expected_coeffs = np.array(
        [0.0 + 0.0j, 0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        dtype=np.complex64,
    )

    np.testing.assert_allclose(coeffs.real, expected_coeffs.real, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(coeffs.imag, expected_coeffs.imag, rtol=1e-5, atol=1e-7)


def test_get_nearby_fourier_bins():

    # Create a sample Fourier transform array
    ft = np.array(
        [
            0 + 0j,
            1 + 0j,
            2 + 0j,
            3 + 0j,
            4 + 0j,
            5 + 0j,
            6 + 0j,
            7 + 0j,
            8 + 0j,
            9 + 0j,
        ],
        dtype=np.complex64,
    )

    # Test with r=4.5 and m=4
    r = 4.5
    m = 4
    bins = get_nearby_fourier_bins(r, ft, m)

    expected_bins = np.array([3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j], dtype=np.complex64)

    np.testing.assert_array_equal(bins.real, expected_bins.real)
    np.testing.assert_array_equal(bins.imag, expected_bins.imag)

    # Test with r=2.2 and m=6
    r = 2.2
    m = 6
    bins = get_nearby_fourier_bins(r, ft, m)

    expected_bins = np.array(
        [0 + 0j, 1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j], dtype=np.complex64
    )

    np.testing.assert_array_equal(bins.real, expected_bins.real)
    np.testing.assert_array_equal(bins.imag, expected_bins.imag)

    # Test with r=3.0 and m=4
    r = 3.0
    m = 4
    bins = get_nearby_fourier_bins(r, ft, m)

    expected_bins = np.array([2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j], dtype=np.complex64)

    np.testing.assert_array_equal(bins.real, expected_bins.real)
    np.testing.assert_array_equal(bins.imag, expected_bins.imag)


def test_fourier_interp():

    r = 12400.55
    N = 32768
    phs = np.pi / 4.0
    signal = np.cos(2 * np.pi * r * np.arange(N) / N + phs)
    ft = np.fft.rfft(signal)

    m = 60
    interp_value = fourier_interp(r, ft, m)

    # Expected value calculated analytically
    expected_value = N / 2 / np.sqrt(2) * complex(1, 1)

    np.testing.assert_allclose(
        interp_value.real, expected_value.real, rtol=1e-2, atol=1e-3
    )
    np.testing.assert_allclose(
        interp_value.imag, expected_value.imag, rtol=1e-2, atol=1e-3
    )

    rs = np.floor(r) + np.linspace(0.0, 1.0, 21)[:-1]
    iv2 = fourier_interp_multi(rs, ft, m)
    assert interp_value.real == iv2[11].real
    assert interp_value.imag == iv2[11].imag

    r = 12400.00
    N = 32768
    phs = np.pi / 4.0
    signal = np.cos(2 * np.pi * r * np.arange(N) / N + phs)
    ft = np.fft.rfft(signal)

    m = 60
    interp_value = fourier_interp(r, ft, m)

    # Expected value calculated analytically
    expected_value = N / 2 / np.sqrt(2) * complex(1, 1)

    np.testing.assert_allclose(
        interp_value.real, expected_value.real, rtol=1e-2, atol=1e-3
    )
    np.testing.assert_allclose(
        interp_value.imag, expected_value.imag, rtol=1e-2, atol=1e-3
    )

    rs = np.floor(r) + np.linspace(0.0, 1.0, 21)[:-1]
    iv2 = fourier_interp_multi(rs, ft, m)
    assert interp_value.real == iv2[0].real
    assert interp_value.imag == iv2[0].imag

    m = 16
    v1 = fourier_interp_multi(rs, ft, m)
    v2 = fourier_interp_FFT(12400, 1, len(rs), ft, m)
    np.testing.assert_allclose(v1.real, v2.real, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(v1.imag, v2.imag, rtol=1e-5, atol=1e-7)
