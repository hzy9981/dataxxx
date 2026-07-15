import scipy.io
import numpy as np
import pandas as pd

data = scipy.io.loadmat('/home/hzy9981/datax/dataX.mat')
X = data['inputtrain'].reshape(500, 500)
y = data['outputtrain'].reshape(500, 500)

print("Correlation with neighbors in 2D:")
# Check correlation of y[i,j] with x[i,j], x[i-1,j], x[i,j-1]
# We'll flatten and use shift
df = pd.DataFrame({'y': y.flatten(), 'x': X.flatten()})
# For i-1,j shift is 500
df['x_up'] = df['x'].shift(500)
df['x_left'] = df['x'].shift(1)
df['y_up'] = df['y'].shift(500)
df['y_left'] = df['y'].shift(1)

print(df.dropna().corr()['y'])
