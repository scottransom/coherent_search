# %%
import numpy as np
import coherent_search.utils as utils
import coherent_search.fourierinterp as fi
import matplotlib.pyplot as plt

# This file has a 10.01234...Hz pulsar signal in it
ft = utils.fftfile("harmonics_hi.fft")

psrf = 10.0123456789123  # Hz
psrr = psrf * ft.T
print(f"psrr = {psrr:.6f}")

m = 32  # Number of Fourier bins to use for interpolation
numbetween = 20  # Number of points to interpolate between Fourier bins

goodr = fi.fourier_interp(psrr, ft.amps, m * 2)
print(f"Good r: {goodr}")
print(f"  power = {np.abs(goodr) ** 2:.6g}")
print(f"  phase = {np.angle(goodr, deg=True):.2f} degrees")


# %%

# Plot the interpolated Fourier amplitudes around the pulsar frequency
numbins = 20
offs = -10
rs = np.floor(psrr) + np.arange(numbins * numbetween) / numbetween + offs
amps = fi.finterp_FFT(int(np.floor(psrr)) + offs, numbins, numbetween, ft.amps, m)

plt.plot(rs - 10000, np.abs(amps) ** 2)
plt.xlabel("Fourier frequency (bins)")
plt.ylabel("Raw Power")
plt.show()

# %%

# Now we will look at accuracy of linear interpolation vs Fourier interpolation
# as a function of 'numbetween', the number "properly" interpolated points
# between the original integer bins

numbetweens = [2, 5, 10, 20, 50]
median_amp_errors = []
median_phs_errors = []
max_amp_errors = []
max_phs_errors = []

# We will consider these the correct interpolated values, since they are
# interpolated using a very large number of Fourier bins (m*2) and a
# large number of points between bins (100)
tamps = fi.finterp_FFT(int(np.floor(psrr)) + offs, numbins, 100, ft.amps, m * 2)
trs = np.floor(psrr) + np.arange(numbins * 100) / 100 + offs

# Only evaluate the error at frequencies where the complex amplitude of
# the true interpolation is > 5% of the median amplitude, to avoid
# dividing by very small numbers and getting huge relative errors
tamps_median = np.median(np.abs(tamps))
tphss = np.unwrap(np.angle(tamps), discont=np.pi, period=2 * np.pi)
mask = np.abs(tamps) > 0.05 * tamps_median

for numbetween in numbetweens:
    rs = np.floor(psrr) + np.arange(numbins * numbetween) / numbetween + offs
    amps = fi.finterp_FFT(int(np.floor(psrr)) + offs, numbins, numbetween, ft.amps, m)
    # Use numpy's 1-D interpolation to get the new interpolated values at the same
    # frequencies as our "true" Fourier interpolation
    interp_amps = np.interp(trs, rs, amps)
    amp_errors = np.abs(np.abs(interp_amps[mask]) - np.abs(tamps[mask])) / np.abs(
        tamps[mask]
    )
    iphss = np.unwrap(np.angle(interp_amps), discont=np.pi, period=2 * np.pi)
    phs_errors = np.abs(iphss[mask] - tphss[mask]) / np.abs(np.angle(tamps[mask]))
    median_amp_errors.append(np.median(amp_errors))
    median_phs_errors.append(np.median(phs_errors))
    max_amp_errors.append(np.max(amp_errors))
    max_phs_errors.append(np.max(phs_errors))

# %%
plt.plot(numbetweens, median_amp_errors, "x-", label="Median Amp Error")
plt.plot(numbetweens, median_phs_errors, "x-", label="Median Phase Error")
plt.plot(numbetweens, max_amp_errors, "x-", label="Max Amp Error")
plt.plot(numbetweens, max_phs_errors, "x-", label="Max Phase Error")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of Points Between Fourier Bins")
plt.ylabel("Relative Error")
plt.title("Interpolation Accuracy vs Number of Points Between Bins")
plt.legend()
plt.grid()
plt.savefig("interp_accuracy.png")
plt.show()
# %%
