# %%
import timeit
import numpy as np
import coherent_search.utils as utils
import coherent_search.fourierinterp as fi
import matplotlib.pyplot as plt

# This file has a 10Hz pulsar signal in it
ft = utils.fftfile("harmonics_hi.fft")

psrf = 10.0  # Hz
psrr = psrf * ft.T

m = 32  # Number of Fourier bins to use for interpolation
numbetween = 20  # Number of points to interpolate between Fourier bins

drs = np.arange(numbetween) / numbetween
multicoeffs = fi.get_finterp_multi_coeffs(drs, m)

minfftlen = utils.next_pow_of_2((m + 1) * numbetween)
maxfftlen = 2**18
fftlens = [2**n for n in range(minfftlen.bit_length(), maxfftlen.bit_length())]

# %%
number = 1000

xt = timeit.timeit(
    lambda: fi.finterp_multi(np.floor(psrr) + drs, ft.amps, m),
    number=number,
)
print(
    f"Rate for un-cached finterp_multi(): {number / xt * numbetween * 1e-6:.2f} Mpts/sec for numbetween={numbetween}, m={m}"
)

# %%

xt = timeit.timeit(
    lambda: fi.finterp_multi(np.floor(psrr) + drs, ft.amps, m, multicoeffs),
    number=number,
)
print(
    f"Rate for    cached finterp_multi(): {number / xt * numbetween * 1e-6:.2f} Mpts/sec for numbetween={numbetween}, m={m}"
)


# %%

ftrates = []

for fftlen in fftlens:
    ftcoeffs = fi.get_finterp_FFT_coeffs(numbetween, m, fftlen)
    numbins = int(np.floor(fftlen / numbetween)) - m
    goodpts = numbins * numbetween
    xt = timeit.timeit(
        lambda: fi.finterp_FFT(
            int(np.floor(psrr)), numbins, numbetween, ft.amps, m, ftcoeffs
        ),
        number=1000,
    )
    ftrate = number / xt * goodpts * 1e-6
    ftrates.append(ftrate)
    print(
        f"Rate for FFT-based interpolation with fftlen={fftlen}: {ftrate:.2f} Mpts/sec"
    )

# %%
if False:
    plt.semilogx(fftlens, ftrates, "x")
    plt.xlabel("FFT Length")
    plt.ylabel("Fourier Interp Rate (Mpts/s)")
    plt.title(f"numbetween={numbetween}, m={m}")
    plt.show()

# %%
barlabs = [rf"$2^{{ {int(np.log2(fftlen))} }}$" for fftlen in fftlens]
plt.bar(barlabs, ftrates, color="red")
plt.xlabel("FFT Length")
plt.ylabel("Fourier Interpolation Rate (Mpts/s)")
plt.title(f"numbetween={numbetween}, m={m}")
plt.savefig(f"fourier_interp_rates_n{numbetween}m{m}.png")
plt.show()

# %%
