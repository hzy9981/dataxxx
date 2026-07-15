import scipy.io
import numpy as np

data = scipy.io.loadmat('/home/hzy9981/datax/dataX.mat')
X = data['inputtrain'].flatten()
y = data['outputtrain'].flatten()

# Bin X and check variance of y in each bin
bins = np.linspace(X.min(), X.max(), 100)
for i in range(len(bins)-1):
    mask = (X >= bins[i]) & (X < bins[i+1])
    if np.sum(mask) > 10:
        print(f"Bin [{bins[i]:.2f}, {bins[i+1]:.2f}]: std(y)={np.std(y[mask]):.4f}, mean(y)={np.mean(y[mask]):.4f}, count={np.sum(mask)}")

# Also check for exact duplicates
from collections import defaultdict
d = defaultdict(list)
for i in range(len(X)):
    d[round(X[i], 5)].append(y[i])

print("\nExact Duplicates check (rounded to 5 decimal places):")
counts = [len(v) for v in d.values() if len(v) > 1]
if counts:
    print(f"Max count for same X: {max(counts)}")
    stds = [np.std(v) for v in d.values() if len(v) > 1]
    print(f"Avg std of y for same X: {np.mean(stds):.4f}")
else:
    print("No duplicates found.")
