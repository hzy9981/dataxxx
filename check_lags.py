import scipy.io
import numpy as np
import pandas as pd

data = scipy.io.loadmat('/home/hzy9981/datax/dataX.mat')
X = data['inputtrain'].flatten()
y = data['outputtrain'].flatten()

df = pd.DataFrame({'x': X, 'y': y})
df['x_prev'] = df['x'].shift(1)
df['x_prev2'] = df['x'].shift(2)
df['y_prev'] = df['y'].shift(1)

print("Correlations:")
print(df.corr())

# Check if y_t can be predicted by x_t, x_{t-1}, ...
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df_clean = df.dropna()
X_lag = df_clean[['x', 'x_prev', 'y_prev']]
y_lag = df_clean['y']

lr = LinearRegression()
lr.fit(X_lag, y_lag)
y_pred = lr.predict(X_lag)
rmse = np.sqrt(mean_squared_error(y_lag, y_pred))
print(f"\nLinear Regression (with lags) RMSE: {rmse:.6f}")
