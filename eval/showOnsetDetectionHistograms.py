import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'out'))

import onsetDetectionValues

# onsetDetectionValues has variables `onsetValues` and `nonOnsetValues`. Print the histograms of both on the same axes.
import matplotlib.pyplot as plt

plt.hist(onsetDetectionValues.onsetValues, bins=50, alpha=0.5, label='Onset Values', density=True)
plt.hist(onsetDetectionValues.nonOnsetValues, bins=50, alpha=0.5, label='Non-Onset Values', density=True)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.title('Onset vs Non-Onset Value Histograms')

plt.show()

print('Max of non-onset values:', max(onsetDetectionValues.nonOnsetValues))
print('Min of onset values:', min(onsetDetectionValues.onsetValues))
# A few false positives are ok, but we don't want any false negatives.
threshold = min(onsetDetectionValues.onsetValues)
print('Suggested threshold for onset detection:', threshold)
print('Percentage of non-onset values above threshold:', sum(1 for v in onsetDetectionValues.nonOnsetValues if v >= threshold) / len(onsetDetectionValues.nonOnsetValues) * 100, '%')