import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'out'))

import matplotlib.pyplot as plt
import numpy as np
import frequencyEstimates as fe

t = [i * fe.secondsPerBlock for i in range(len(fe.frequencyEstimates))]
plt.subplot(2, 1, 1)
plt.plot(t, 1200 * np.log2(np.array(fe.frequencyEstimates)/440))
plt.xlabel('t (s)')
plt.ylabel('Frequency Estimate (cents relative to A4)')
plt.suptitle('Frequency Estimates and Onset Detection over Time')
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(t, fe.onsets)
plt.xlabel('t (s)')
plt.ylabel('Onset Detected (1 or 0)')
plt.suptitle('Frequency Estimates and Onset Detection over Time')
plt.grid()

plt.gcf().canvas.manager.set_window_title("Frequency Estimates over Time")
plt.show()
