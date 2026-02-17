# Show the magnitude response of a filter with differential equation y[n] = b0 * x[n] + b1 * x[n-1] - a1 * y[n-2]
# for b0 = 0.124707982, b1 = -0.124707982, a1 = -0.875292003

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, freqz

# Design a first-order Butterworth high-pass filter
fs = 44100  # Sampling frequency
fc = 5000   # Cutoff frequency (Hz)
order = 2
nyq = 0.5 * fs
normal_cutoff = fc / nyq
b, a = butter(order, normal_cutoff, btype='low', analog=False)

# Frequency response
w, h = freqz(b, a, worN=512)
plt.figure()
plt.plot(w / np.pi * fs / 2, 20 * np.log10(np.abs(h)))
plt.title('Magnitude Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.xlim(0, 20000)
plt.ylim(-60, 5)

plt.savefig('lowpass_filter_response.png')
plt.close()

print("b coefficients:", ", ".join([str(coef) for coef in b]))
print("a coefficients:", ", ".join([str(coef) for coef in a]))