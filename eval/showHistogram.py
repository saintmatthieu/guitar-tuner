import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'out'))

import errors
import matplotlib.pyplot as plt
import numpy as np

# bins go from -2500 to 2500 in buckets of 100, centered at 0
bins = [i for i in range(-2550, 2600, 100)]
counts, bins, _ = plt.hist(errors.errors, bins=bins, edgecolor='black')
plt.clf()  # Clear the previous plot

weights = np.ones_like(errors.errors) / len(errors.errors) * 100
plt.hist(errors.errors, bins=bins, edgecolor='black', weights=weights)

counts, bin_edges = np.histogram(errors.errors, bins=bins, weights=weights)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
zero_bin_index = np.argmin(np.abs(bin_centers))
percentage_at_zero = counts[zero_bin_index]

plt.figure(1)
plt.xlabel('Error (Cents)')
plt.ylabel('Percents (%)')
plt.xlim(-2500, 2500)
# ticks every octave (1200 cents)
plt.xticks([-2400, -1200, 0, 1200, 2400], ['-2 Octaves', '-1 Octave', '0', '1 Octave', '2 Octaves'])
plt.title('Error histogram : {:.1f}% within +/- 50 cents'.format(percentage_at_zero))
plt.grid(True)
plt.gcf().canvas.manager.set_window_title("Error histogram")

# # meaningful if only one file was tested
# plt.figure(2)
# plt.plot(errors.errors)
# plt.xlabel('Frame Index')
# plt.ylabel('Error (Cents)')
# plt.title('Error over time')
# plt.grid(True)
# plt.gcf().canvas.manager.set_window_title("Error over time")

plt.show()