import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'out'))

import matplotlib.pyplot as plt
import numpy as np
import frequencyEstimates as fe

t = [i * fe.secondsPerBlock for i in range(len(fe.frequencyEstimates))]
plt.plot(t, 1200 * np.log2(np.array(fe.frequencyEstimates)/440))
plt.xlabel('t (s)')
plt.ylabel('Frequency Estimate (cents relative to A4)')
plt.title('Frequency Estimates over Time')
plt.grid()
plt.gcf().canvas.manager.set_window_title("Frequency Estimates over Time")
plt.show()
