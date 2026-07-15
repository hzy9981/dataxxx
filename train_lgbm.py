import scipy.io
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

data = scipy.io.loadmat('/home/hzy9981/datax/dataX.mat')
X_train = data['inputtrain']
y_train = data['outputtrain'].flatten()
X_test = data['inputtest']
y_test = data['outputtest'].flatten()

model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=127, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"LightGBM RMSE: {rmse:.6f}")
