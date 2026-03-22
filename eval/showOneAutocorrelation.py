import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'out'))

import PitchDetectorLog as pdl
import matplotlib.pyplot as plt

t0 = pdl.audioIndex / pdl.sampleRate
t = np.array([i / pdl.sampleRate for i in range(len(pdl.inputAudio))])
fig = 0

# Write the windowed audio to file:
from scipy.io import wavfile
windowedAudio = np.array(pdl.windowedAudio).astype(np.float32)
normalizedWindowedAudio = windowedAudio / np.max(np.abs(windowedAudio)) * 0.9
wavfile.write(os.path.join(os.path.dirname(__file__), 'out', 'windowedAudio.wav'), pdl.sampleRate, normalizedWindowedAudio)

fig += 1
plt.figure(fig)
plt.plot(t, pdl.cepstrum)
plt.grid(True)
plt.title("Cepstrum")
plt.gcf().canvas.manager.set_window_title("Cepstrum")

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
plt.plot(t, pdl.xcorr)
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(pdl.xcorr)
plt.grid(True)
plt.suptitle("Autocorrelation")
plt.gcf().canvas.manager.set_window_title("Autocorrelation")

plt.show()