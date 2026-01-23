# A script that takes an input python file.
# This file has a list of frequency estimation results: `results = [r0, r1, r2, ...]` (without line breaks)
# Eliminate zeros from this list:

import sys
import math
import matplotlib.pyplot as plt
import os

input_dir = sys.argv[1]
sr = 44100
blockSize = 512

# for each file with extension .py in input_dir
for filename in os.listdir(input_dir):
    if not filename.endswith('.py'):
        continue
    input_file = os.path.join(input_dir, filename)

    with open(input_file, 'r') as f:
        content = f.read()
        # interpret as python list
        exec(content)

    t = [i * blockSize / sr for i in range(len(results))]
    non_zero_indices = [i for i, r in enumerate(results) if r != 0]

    E3 = 164.813778456435
    deviations_cents = [1200 * math.log2(r / E3) for r in results if r != 0]

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot([t[i] for i in non_zero_indices], deviations_cents)
    plt.xlabel('time (s)')
    plt.ylabel('Deviation from E3 (cents)')
    plt.grid(True)

    # plot histogram
    plt.subplot(2, 1, 2)
    plt.hist(deviations_cents, bins=50)
    plt.xlabel('Deviation from E3 (cents)')
    plt.ylabel('Count')

    # add file title as the plot title
    plt.suptitle(f'{filename} - max dev: {max(deviations_cents) - min(deviations_cents):.2f} cents')

    # save the plot as a png file
    output_png = os.path.join(input_dir, filename + '_deviation.png')
    plt.savefig(output_png)

plt.show()

