import scipy.io
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = scipy.io.loadmat('/home/hzy9981/datax/dataX.mat')
X_train = data['inputtrain']
y_train = data['outputtrain']
X_test = data['inputtest']
y_test = data['outputtest']

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Linear Regression RMSE: {rmse:.6f}")
print(f"Coefficients: {lr.coef_}")
print(f"Intercept: {lr.intercept_}")
