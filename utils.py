import os
import numpy as np
from pathlib import Path
from typing import Union


class simpleinf:
    "A simple PRESTO .inf file reader (only key params)"

    def __init__(self, inf: Union[str, os.PathLike]) -> None:
        """Initialize a PRESTO .inf file class instance.

        Parameters
        ----------
        inf : file or str or Path
            The PRESTO .inf file to open
        """
        self.inf: os.PathLike = inf if isinstance(inf, os.PathLike) else Path(inf)
        try:
            with open(self.inf, "r") as file:
                for line in file:
                    if line.startswith(" Object being observed"):
                        self.object = line.split("=")[-1].strip()
                        continue
                    if line.startswith(" Epoch"):
                        self.epoch = float(line.split("=")[-1].strip())
                        continue
                    if line.startswith(" Number of bins"):
                        self.N = int(line.split("=")[-1].strip())
                        continue
                    if line.startswith(" Width of each time series bin"):
                        self.dt = float(line.split("=")[-1].strip())
                        continue
                    if line.startswith(" Dispersion measure"):
                        self.DM = float(line.split("=")[-1].strip())
                        continue
        except FileNotFoundError:
            print(f"Error: The .inf file '{self.inf}' was not found.")


class fftfile:
    "A PRESTO FFT file (i.e. with suffix '.fft') and associated metadata"

    def __init__(self, ff: Union[str, os.PathLike]) -> None:
        """Initialize a PRESTO fftfile class instance.

        Parameters
        ----------
        ff : file or str or Path
            The PRESTO .fft file to open
        """
        self.ff: os.PathLike = ff if isinstance(ff, os.PathLike) else Path(ff)
        self.amps = np.memmap(self.ff, dtype=np.complex64)
        self.inf = simpleinf(f"{str(self.ff)[:-4]}.inf")
        self.N: int = self.inf.N
        self.T: float = self.N * self.inf.dt
        self.dereddened = True if "_red.fft" in str(self.ff) else False
        self.detrended = True if self.dereddened else False
        self.DC, self.Nyquist = self.amps[0].real, self.amps[0].imag
        self.df: float = 1.0 / self.T

    @property
    def freqs(self) -> np.ndarray:
        """The frequencies (in Hz) for the FFT amplitudes."""
        self._freqs = np.linspace(0.0, self.N // 2 * self.df, self.N // 2)
        return self._freqs


def next_pow_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    return 1 << (n - 1).bit_length()
