import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'out'))

import roc_curve as roc
import matplotlib.pyplot as plt

plt.plot(roc.falsePositiveRates, roc.truePositiveRates)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.title('ROC Curve: AUC = {:.4f}, threshold for allowed FPR of {:.1f}%: {:.4f}'.format(roc.AUC, roc.allowedFalsePositiveRate * 100, roc.threshold))
plt.show()
