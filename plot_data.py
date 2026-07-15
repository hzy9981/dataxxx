import scipy.io
import matplotlib.pyplot as plt

data = scipy.io.loadmat('dataX.mat')
X_train = data['inputtrain']
y_train = data['outputtrain'].ravel()

# Plot first 1000 points
plt.scatter(X_train[:1000], y_train[:1000], s=1)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Subset of Data')
plt.savefig('data_plot.png')
print("Plot saved to data_plot.png")
