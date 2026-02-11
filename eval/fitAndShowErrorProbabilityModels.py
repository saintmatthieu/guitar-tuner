import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'out'))

import errors
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Show two histograms, one for the confidence scores for octaviated estimates (error beyond +/- 50 cents), and one for those of non-octaviated estimates.
within_range_scores = np.array([score for error, score in zip(errors.errors, errors.scores) if -50 <= error <= 50 and score > 0.01])
outside_range_scores = np.array([score for error, score in zip(errors.errors, errors.scores) if (error < -50 or error > 50) and score > 0.01])

# Fit Beta distribution for within-range (bounded on [0,1], good for data near boundaries)
# Clamp values slightly away from exact 1.0 for numerical stability
within_clamped = np.clip(within_range_scores, 0.001, 0.999)
within_params = stats.beta.fit(within_clamped, floc=0, fscale=1)
# Fit skew normal for outside-range (octaviated)
outside_params = stats.skewnorm.fit(outside_range_scores)

print(f"Within range Beta params (a, b): {within_params[:2]}")
print(f"Outside range skew normal params (a, loc, scale): {outside_params}")

# Print C++ constants
print("\n// C++ constants for fitted distributions:")
print(f"constexpr double kBetaA = {within_params[0]};")
print(f"constexpr double kBetaB = {within_params[1]};")
print(f"constexpr double kSkewA = {outside_params[0]};")
print(f"constexpr double kSkewLoc = {outside_params[1]};")
print(f"constexpr double kSkewScale = {outside_params[2]};")

# Create fitted distributions for PDF evaluation
within_dist = stats.beta(within_params[0], within_params[1])
outside_dist = stats.skewnorm(*outside_params)

# Plot histograms
plt.hist(within_range_scores, bins=100, alpha=0.5, label='Within +/- 50 cents', density=True)
plt.hist(outside_range_scores, bins=100, alpha=0.5, label='Outside +/- 50 cents', density=True)

# Generate x values for PDF plots
x_min = min(within_range_scores.min(), outside_range_scores.min())
x_max = max(within_range_scores.max(), outside_range_scores.max())
x = np.linspace(x_min, x_max, 200)

# Plot PDFs
plt.plot(x, within_dist.pdf(x), 'b-', linewidth=2, label='Within fit (Beta)')
plt.plot(x, outside_dist.pdf(x), 'r-', linewidth=2, label='Outside fit (skew normal)')

plt.xlabel('Confidence Score')
plt.ylabel('Density')
plt.title('Distributions of confidence scores for octaviated vs non-octaviated auto-correlation estimates')
plt.legend()

# New plot: histogram of all scores
all_scores = np.array([score for score in errors.scores if score > 0.01])
plt.figure()
plt.hist(all_scores, bins=100, alpha=0.7, density=True)

# Mixture model: combine the two fitted distributions with mixing weights
n_within = len(within_range_scores)
n_outside = len(outside_range_scores)
w_within = n_within / (n_within + n_outside)
w_outside = n_outside / (n_within + n_outside)

print(f"Mixture weights: within={w_within:.3f}, outside={w_outside:.3f}")
print(f"constexpr double kPriorGood = {w_within};")

# Plot mixture PDF
x_all = np.linspace(0.01, 1.0, 200)
mixture_pdf = w_within * within_dist.pdf(x_all) + w_outside * outside_dist.pdf(x_all)
plt.plot(x_all, mixture_pdf, 'k-', linewidth=2, label=f'Mixture (w={w_within:.2f}/{w_outside:.2f})')

plt.xlabel('Confidence Score')
plt.ylabel('Density')
plt.title('Distribution of all confidence scores')
plt.legend()

plt.show()