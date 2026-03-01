# %%
import numpy as np
import coherent_search.utils as utils
import coherent_search.fourierinterp as fi
import matplotlib.pyplot as plt

# This file has a 10.0123456789123 Hz pulsar signal in it
ft = utils.fftfile("harmonics_hi_red.fft")

psrf = 10.0123456789123  # Hz
psrr = psrf * ft.T

m = 32  # Number of Fourier bins to use for interpolation
numbetween = 20  # Number of points to interpolate between Fourier bins

# Number of harmonics for coherent harmonic fold (*2 for number of profile bins)
chharms = 32

# search range (at fundamental) in Fourier bins for coherent harmonic fold
searchr = 4
lors = np.floor(psrr) - searchr // 2 + np.arange(searchr * 2 * chharms) / (2 * chharms)

# We will load the interpolated Fourier amplitudes into this array
ftprofs = np.zeros((searchr * 2 * chharms, chharms + 1), dtype=np.complex128)

# loop through the harmonics
for ii in range(1, chharms + 1):
    rs = lors * ii
    lobin = int(np.floor(rs[0]))
    hibin = int(np.ceil(rs[-1])) + 1
    numbins = hibin - lobin
    if hibin < ft.N // 2:
        print(
            f"Harmonic {ii} covers {rs[-1] - rs[0]:.2f} bins (lobin={lobin}, hibin={hibin}), numbins={numbins}"
        )
        tmpamps = fi.finterp_FFT(lobin, numbins, numbetween, ft.amps, m)
        tmprs = np.arange(numbins * numbetween) / numbetween + lobin
        ftprofs[:, ii] = np.interp(rs, tmprs, tmpamps)
    else:
        print(f"Harmonic {ii} is above Nyquist frequency. Skipping.")

# %%
# Inverse FFT the interpolated Fourier amplitudes to get the pulse profiles at each trial frequency
profs = np.fft.irfft(ftprofs, axis=1)

maxmetric = np.max(profs, axis=1)
plt.plot(lors - 10000, maxmetric)
plt.xlabel("Fourier Frequency - 10000 (bins)")
plt.ylabel("Max Profile Value")
plt.show()

# %%
psrindex = np.argmin(np.abs(lors - psrr))
print(f"Best profile is at index {np.argmax(maxmetric)}, should be {psrindex}")

# %%
plt.plot(profs[np.argmax(maxmetric)])
plt.xlabel("Profile Bin")
plt.ylabel("Intensity")
plt.show()

# %%
