import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'out'))

import spectrum
import matplotlib.pyplot as plt
plt.plot(spectrum.bins, spectrum.rectangular, label="Rectangular")
plt.plot(spectrum.bins, spectrum.hann, label="Hann")
plt.plot(spectrum.bins, spectrum.hamming, label="Hamming")
plt.plot(spectrum.bins, spectrum.minimumThreeTerm, label="Minimum Three-Term")
plt.title("Spectrum")
plt.xlabel("Frequency Bin")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.show()