import scipy.io
import numpy as np

data = scipy.io.loadmat('/home/hzy9981/datax/dataX.mat')
X = data['inputtrain'].flatten()
y = data['outputtrain'].flatten()

print("Correlations with shifted X:")
for lag in range(-5, 6):
    if lag < 0:
        corr = np.corrcoef(X[-lag:], y[:lag])[0,1]
    elif lag > 0:
        corr = np.corrcoef(X[:-lag], y[lag:])[0,1]
    else:
        corr = np.corrcoef(X, y)[0,1]
    print(f"Lag {lag}: {corr:.6f}")

print("\nCorrelations with shifted y:")
for lag in range(1, 6):
    corr = np.corrcoef(y[:-lag], y[lag:])[0,1]
    print(f"y Lag {lag}: {corr:.6f}")
