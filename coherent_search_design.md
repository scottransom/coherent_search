# Coherent Search Design Notes

 - Original Python repo: `/home/sransom/git/coherent_search`
 - New Julia repo: `/home/sransom/git/CoherentSearch.jl`

The following are ideas for the implementation of a "coherent" pulsar search which uses the complex Fourier amplitudes from the FFT of a long time series (typically 1M-1G points) to build up the complex harmonics of candidate pulsar signals at each possible spin frequency, which are then inverse-FFT'd into a candidate pulse shape that is tested by some metric for its significance and "pulsar-ness".

To get an accurate pulse shape reconstruction, each harmonic `h` must be measured at a very precise Fourier frequency `r_h = r_f * h`, where `r_f` is the fundamental spin frequency of the pulsar, which will require interpolation between the integer Fourier frequency bins from the original FFT. The interpolated Fourier phase for each harmonic must be correct to roughly `2*pi/Nbins` in radians, where `Nbins` is the number of bins in the reconstructed pulse profile. In general, we will need `Nharm = Nbins/2` harmonics to be computed via Fourier interpolation, with the "zero" bin set to 0.0 (that means that the short complex->real inverse FFT will be of length `Nbins/2+1`).

If the original long FFT has an integer bin Fourier frequency resolution of `1/T`, where `T` was the duration of the original time series, then we will step through the original FFT in steps of `0.5/T` frequency *for the highest harmonic of each candidate at frequency `r_f`*. In other words, for `h = Nharm`, our `deltar = 0.5/T` or 0.5 of an integer bin. And so, for the harmonics `h`, the stepsize is `deltar_h = 0.5 * n / (T * Nharm)`, where `h` goes from 1 to Nharm. This means that we need finer interpolated Fourier frequency resolution at the lower numbered harmonics.

Once we have a reconstructed pulse profile, we compute its significance via some metric (a very simple one which works for pulsars with narrow pulse profiles is simply the max value of the profile divided by the median of the profile), and record it as a good candidate if that metric passes some threshold. Finally, we remove duplicate and/or harmonically-related candidates and return the candidate list in sorted order by the metric.

## Ideas for speed improvements

Fourier interpolation of many points at once can be handled by Fourier-domain techniques using convolution/correlation theorems and a sinc + complex phase based kernel. In the new Julia-based repo CoherentSearch.jl, this is performed by the routine `finterp_fft()`, where the resulting Fourier resolution in bins is `1/numbetween`. Since we have end-effects due to the correlation that must be chopped off, we usually want the FFT size `fftlen` to be a power-of-two (for efficiency) of length much greater than `numbetween` so that we get many good interpolated points out of each call. I suspect values of 1024 < `fftlen` < 65536 will provide the most interpolation throughput, but we will need to test this. Those points can be further interpolated via linear interpolation as long as the `numbetween` is large enough (probably >= 16) so that we meet our phase accuracy requirements. This means that we want to cache our FFTW plans for forward and inverse FFTs of `fftlen`, as well as the sinc coefficients, whenever possible.

Similarly, the plan for a complex->real inverse FFT of length `Nbins/2+1` should be cached for computing our reconstructed pulse profiles.

Both of these speed improvements should be implemented.

## Ideas for parallelization

Since we can efficiently Fourier interpolate a grid of dense values using `finterp_fft()` or `finterp_fft()` plus linear interpolation, we should be generating *many* values for each harmonic `h` around the Fourier frequencies `r_h` every time we call those interpolation routines.

I propose that we break the code up into two main parts organized within a loop over the full length of the input FFT, where we loop over each harmonic `h` to create many (`Nprof` in thousands or tens of thousands) FFTs of the candidate profiles using `finterp_fft()` for each harmonic, and then FFT and search all `Nprof` of those candidates in a different sub-loop.  The structure would be something like the following:

- loop #1 over `lofreq` to `hifreq` in the input array:
  - loop #2 from 1 to `Nharm`:
    - Generate `Nprof` amplitudes for `h` and store as rows in `h`x`Nprof` 2D array
  - loop #3 from 1 to `Nprof`:
    - inverse FFT each profile
    - compute metric for candidate profile and store if > threshold

For loop #2, since each harmonic `h` is independent, we can have a separate thread use cached FFT plans, interpolation coefficients, and possibly different `numbetween` values as appropriate for the `deltar_h` needed to independently fill the shared `h`x`Nprof` 2D array. If sharing this array is not possible, we can make an `Nprof`-length vector for each harmonic which we merge into a 2D array at the end of the loop. There would need to be a set of cached information for each `h`, including `fftlen`, `numbetween`, starting and ending `r`, FFTW plan, and interpolation coefficients. I think that could be done via an array of structures.

Since we can be using different length FFTs and different `numbetween` values for each harmonic, we can tune each harmonic to give the best Fourier interpolation throughput possible. We should do this via a separate script that benchmarks throughput as a function of `fftlen` and `numbetween`. Something like this already exists in the original Python repo for coherent_search in `examples/speed_test.py`.

For loop #3, the FFTs and the metric checks are all independent. The only thing that is tricky is that we need to be careful about storing and merging candidates. Maybe each thread keeps its own list of candidates which are merged at the end of loop #1.
