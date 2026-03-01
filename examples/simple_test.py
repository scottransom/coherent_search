import numpy as np
import coherent_search.fourierinterp as fi
import matplotlib.pyplot as plt

r = 12400.55
N = 32768
phs = np.pi / 4.0
signal = np.cos(2 * np.pi * r * np.arange(N) / N + phs)
ft = np.fft.rfft(signal)

rs = np.floor(r) + np.linspace(0.0, 1.0, 21)[:-1]

ms = [60, 30, 16, 12, 10, 8, 6, 4, 2]
goodx = fi.fourier_interp(r, ft, 100)
print(f"good val = {np.abs(goodx)}")

for m in ms:
    y = fi.finterp_multi(rs, ft, m)
    # plt.plot(rs - 12400, np.abs(y), label=f"m = {m}")
    fe = (np.abs(goodx) - np.abs(y[11])) / np.abs(goodx) * 100
    print(m, np.abs(y[11]), f"frac err = {fe:.2f}%")
    # print(y[11])

# plt.legend()
# plt.show()


m = 16
# y = fi.finterp_multi(rs, ft, m)
# fy = fi.finterp_FFT(12400, 1, 20, ft, m)
fy = fi.finterp_FFT(12400 - 10, 20, 20, ft, m)
rs = 12400 - 10 + np.arange(20 * 20) / 20
# print(y)
# print(fy)
plt.plot(rs, np.abs(fy))
plt.show()
