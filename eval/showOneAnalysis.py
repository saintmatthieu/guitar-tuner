import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'out'))

import PitchDetectorLog as pdl
import matplotlib.pyplot as plt

t0 = pdl.audioIndex / pdl.sampleRate
t = np.array([i / pdl.sampleRate for i in range(len(pdl.inputAudio))])
cepstrumT = [i / pdl.sampleRate for i in range(len(pdl.cepstrum))]
hzPerBin = pdl.sampleRate / pdl.fftSize
F = len(pdl.windowedDbSpec) // 2
f = [i * hzPerBin for i in range(F)]
fig = 0

fig += 1
plt.figure(fig)
plt.plot(t, pdl.inputAudio)
plt.grid(True)
plt.title("Input Audio Signal")
plt.gcf().canvas.manager.set_window_title("Input Audio Signal")

fig += 1
plt.figure(fig)
plt.plot(t, pdl.windowedAudio)
plt.grid(True)
plt.title("Windowed Audio Signal")
plt.gcf().canvas.manager.set_window_title("Windowed Audio Signal")

fig += 1
plt.figure(fig)
plt.subplot(2, 1, 1)
plt.plot(f, pdl.dbSpectrum[:F])
plt.xlabel("Frequency (Hz)")
plt.ylabel("dB")
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(pdl.dbSpectrum[:F])
plt.xlabel("Frequency Bin")
plt.ylabel("dB")
plt.grid(True)
# plt.xlim(0, 2000 / hzPerBin)
plt.suptitle("dB Spectrum")
plt.gcf().canvas.manager.set_window_title("dB Spectrum")

fig += 1
plt.figure(fig)
plt.subplot(2, 1, 1)
plt.plot(t, pdl.xcorr)
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(pdl.xcorr)
plt.grid(True)
plt.suptitle("Autocorrelation")
plt.gcf().canvas.manager.set_window_title("Autocorrelation")

fig += 1
plt.figure(fig)
plt.subplot(2, 1, 1)
plt.plot(f, pdl.windowedDbSpec[:F])
plt.xlabel("Frequency (Hz)")
plt.ylabel("dB")
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(pdl.windowedDbSpec[:F])
plt.xlabel("Frequency Bin")
plt.ylabel("dB")
plt.grid(True)
# plt.xlim(0, 2000 / hzPerBin)
plt.suptitle("Windowed dB Spectrum")
plt.gcf().canvas.manager.set_window_title("Windowed dB Spectrum")

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
plt.gcf().canvas.manager.set_window_title("Cepstrum")

fig += 1
plt.figure(fig)
plt.subplot(2, 1, 1)
plt.plot(f, pdl.spectrumEnvelope[:F])
plt.xlabel("Frequency (Hz)")
plt.ylabel("dB")
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(pdl.spectrumEnvelope[:F])
plt.xlabel("Frequency Bin")
plt.ylabel("dB")
plt.grid(True)
plt.suptitle("Spectral Envelope")
plt.gcf().canvas.manager.set_window_title("Spectral Envelope")

fig += 1
plt.figure(fig)
plt.subplot(2, 1, 1)
plt.plot(f, pdl.idealSpectrum[:F])
plt.xlabel("Frequency (Hz)")
plt.ylabel("dB")
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(pdl.idealSpectrum[:F])
plt.xlabel("Frequency Bin")
plt.ylabel("dB")
plt.grid(True)
plt.suptitle("Ideal Spectrum")
plt.gcf().canvas.manager.set_window_title("Ideal Spectrum (anything below 0 is noise)")

# fig += 1
# plt.figure(fig)
# for harmonic in range(pdl.hpsNumHarmonics):
#   plt.subplot(pdl.hpsNumHarmonics, 1, harmonic + 1)
#   varName = "hpsAfterHarmonic" + str(harmonic + 1)
#   hps = getattr(pdl, varName)
#   hps = [x / max(hps) for x in hps]
#   plt.plot(f[:len(hps)], hps)
#   if harmonic < pdl.hpsNumHarmonics - 1:
#     varName = "hpsDownsampledHarmonic" + str(harmonic + 2)
#     downsampled = getattr(pdl, varName)
#     downsampled = [x / max(downsampled) for x in downsampled]
#     plt.plot(f[:len(downsampled)], downsampled, linestyle='dashed')
#   plt.xlabel("Frequency (Hz)")
#   plt.ylabel("HPS Product")
# plt.grid(True)
# plt.suptitle("Harmonic Product Spectrum")
# plt.gcf().canvas.manager.set_window_title("Harmonic Product Spectrum")

plt.show()