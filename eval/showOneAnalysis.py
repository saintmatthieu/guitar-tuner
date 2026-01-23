import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'out'))

import FormantShifterLog as fsl
import matplotlib.pyplot as plt

fig = 1
t = [i / fsl.sampleRate for i in range(len(fsl.inputAudio))]
cepstrumT = [i / fsl.sampleRate for i in range(len(fsl.cepstrum))]

plt.figure(fig)
plt.plot(t, fsl.inputAudio)
plt.grid(True)
plt.title("Input Audio Signal")

fig += 1
plt.figure(fig)
plt.plot(t, fsl.windowedAudio)
plt.grid(True)
plt.title("Windowed Audio Signal")

fig += 1
plt.figure(fig)
plt.subplot(2, 1, 1)
plt.plot(t, fsl.xcorr)
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(fsl.xcorr)
plt.grid(True)
plt.suptitle("Autocorrelation")

fig += 1
plt.figure(fig)
hzPerBin = fsl.sampleRate / fsl.fftSize
F = len(fsl.logMagSpectrum) // 2
f = [i * hzPerBin for i in range(F)]
plt.subplot(2, 1, 1)
plt.plot(f, fsl.logMagSpectrum[:F])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Log Magnitude")
plt.grid(True)
# plt.xlim(0, 2000)
plt.subplot(2, 1, 2)
plt.plot(fsl.logMagSpectrum[:F])
plt.xlabel("Frequency Bin")
plt.ylabel("Log Magnitude")
plt.grid(True)
# plt.xlim(0, 2000 / hzPerBin)
plt.suptitle("Log Magnitude Spectrum")

fig += 1
plt.figure(fig)
plt.subplot(2, 1, 1)
plt.plot(cepstrumT, fsl.cepstrum)
plt.xlabel("Quefrency (s)")
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(fsl.cepstrum)
plt.xlabel("Cepstrum Bin")
plt.grid(True)
plt.suptitle("Cepstrum")
plt.show()