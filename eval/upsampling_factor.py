import numpy as np

factor = 4
sr = 48000 * factor
E4_freq = 329.63
E4_period_samples = sr / E4_freq
E4_floor_freq = sr / np.floor(E4_period_samples)
E4_ceil_freq = sr / np.ceil(E4_period_samples)
halfwayFreq = (E4_floor_freq + E4_ceil_freq) / 2
max_err_cents = 1200 * np.log2(E4_floor_freq / halfwayFreq)

print(f"max error: {max_err_cents:.2f} cents")