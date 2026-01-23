import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'out'))

import PitchDetectorLog as pdl
import matplotlib.pyplot as plt

fig = 1
t = [i / pdl.sampleRate for i in range(len(pdl.inputAudio))]
cepstrumT = [i / pdl.sampleRate for i in range(len(pdl.cepstrum))]

plt.figure(fig)
plt.plot(t, pdl.inputAudio)
plt.grid(True)
plt.title("Input Audio Signal")

fig += 1
plt.figure(fig)
plt.plot(t, pdl.windowedAudio)
plt.grid(True)
plt.title("Windowed Audio Signal")

fig += 1
plt.figure(fig)
plt.subplot(2, 1, 1)
plt.plot(t, pdl.xcorr)
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(pdl.xcorr)
plt.grid(True)
plt.suptitle("Autocorrelation")

fig += 1
plt.figure(fig)
hzPerBin = pdl.sampleRate / pdl.fftSize
F = len(pdl.logMagSpectrum) // 2
f = [i * hzPerBin for i in range(F)]
plt.subplot(2, 1, 1)
plt.plot(f, pdl.logMagSpectrum[:F])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Log Magnitude")
plt.grid(True)
# plt.xlim(0, 2000)
plt.subplot(2, 1, 2)
plt.plot(pdl.logMagSpectrum[:F])
plt.xlabel("Frequency Bin")
plt.ylabel("Log Magnitude")
plt.grid(True)
# plt.xlim(0, 2000 / hzPerBin)
plt.suptitle("Log Magnitude Spectrum")

fig += 1
plt.figure(fig)
plt.subplot(2, 1, 1)
plt.plot(cepstrumT, pdl.cepstrum)
plt.xlabel("Quefrency (s)")
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(pdl.cepstrum)
plt.xlabel("Cepstrum Bin")
plt.grid(True)
plt.suptitle("Cepstrum")
plt.show()