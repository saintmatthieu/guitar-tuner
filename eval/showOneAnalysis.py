import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'out'))

import PitchDetectorLog as pdl
import matplotlib.pyplot as plt

# configure matplotlib to dark mode
plt.style.use('dark_background')

t0 = pdl.audioIndex / pdl.sampleRate
t = np.array([i / pdl.sampleRate for i in range(len(pdl.inputAudio))])
cepstrumT = [i / pdl.sampleRate for i in range(len(pdl.cepstrum))]
hzPerBin = pdl.sampleRate / pdl.fftSize
F = len(pdl.logMagSpectrum) // 2
f = [i * hzPerBin for i in range(F)]
fig = 0

fig += 1
plt.figure(fig)
plt.plot(t + t0, pdl.inputAudio)
plt.grid(False)
plt.title("Input Audio Signal")

fig += 1
plt.figure(fig)
plt.plot(t + t0, pdl.windowedAudio)
plt.grid(False)
plt.title("Windowed Audio Signal")

fig += 1
plt.figure(fig)
plt.subplot(2, 1, 1)
plt.plot(t, pdl.xcorr)
plt.grid(False)
plt.subplot(2, 1, 2)
plt.plot(pdl.xcorr)
plt.grid(False)
plt.suptitle("Autocorrelation")

fig += 1
plt.figure(fig)
plt.subplot(2, 1, 1)
plt.plot(f, pdl.logMagSpectrum[:F])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Log Magnitude")
plt.grid(False)
# plt.xlim(0, 2000)
plt.subplot(2, 1, 2)
plt.plot(pdl.logMagSpectrum[:F])
plt.xlabel("Frequency Bin")
plt.ylabel("Log Magnitude")
plt.grid(False)
# plt.xlim(0, 2000 / hzPerBin)
plt.suptitle("Log Magnitude Spectrum")

fig += 1
plt.figure(fig)
plt.subplot(2, 1, 1)
plt.plot(cepstrumT, pdl.cepstrum)
plt.xlabel("Quefrency (s)")
plt.grid(False)
plt.subplot(2, 1, 2)
plt.plot(pdl.cepstrum)
plt.xlabel("Cepstrum Bin")
plt.grid(False)
plt.suptitle("Cepstrum")

fig += 1
plt.figure(fig)
plt.subplot(2, 1, 1)
plt.plot(cepstrumT, pdl.cepstrumFiltered)
plt.xlabel("Time (s)")
plt.grid(False)
plt.subplot(2, 1, 2)
plt.plot(pdl.cepstrumFiltered)
plt.xlabel("Cepstrum Bin")
plt.grid(False)
plt.suptitle("Filtered Cepstrum")

fig += 1
plt.figure(fig)
for harmonic in range(pdl.hpsNumHarmonics):
  plt.subplot(pdl.hpsNumHarmonics, 1, harmonic + 1)
  varName = "hpsAfterHarmonic" + str(harmonic + 1)
  hps = getattr(pdl, varName)
  hps = [x / max(hps) for x in hps]
  plt.plot(f[:len(hps)], hps)
  if harmonic < pdl.hpsNumHarmonics - 1:
    varName = "hpsDownsampledHarmonic" + str(harmonic + 2)
    downsampled = getattr(pdl, varName)
    downsampled = [x / max(downsampled) for x in downsampled]
    plt.plot(f[:len(downsampled)], downsampled, linestyle='dashed')
  plt.xlabel("Frequency (Hz)")
  plt.ylabel("HPS Product")
plt.grid(False)
plt.suptitle("Harmonic Product Spectrum")

plt.show()