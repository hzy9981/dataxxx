import scipy.io
import numpy as np

data = scipy.io.loadmat('/home/hzy9981/datax/dataX.mat')
X = data['inputtrain'].flatten()

# Split X into 5 columns
X5 = X.reshape(-1, 5)
print("Split into 5 columns:")
for i in range(5):
    print(f"Col {i}: mean={np.mean(X5[:,i]):.4f}, std={np.std(X5[:,i]):.4f}")

# Split X into 2 columns
X2 = X.reshape(-1, 2)
print("\nSplit into 2 columns:")
for i in range(2):
    print(f"Col {i}: mean={np.mean(X2[:,i]):.4f}, std={np.std(X2[:,i]):.4f}")

# Split X into 10 columns
X10 = X.reshape(-1, 10)
print("\nSplit into 10 columns:")
for i in range(10):
    print(f"Col {i}: mean={np.mean(X10[:,i]):.4f}, std={np.std(X10[:,i]):.4f}")
