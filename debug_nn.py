import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
data = scipy.io.loadmat('/home/hzy9981/datax/dataX.mat')
X_train = data['inputtrain']
y_train = data['outputtrain']
X_test = data['inputtest']
y_test = data['outputtest']

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test)

# Build model
model = keras.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(X_train_scaled, y_train_scaled, 
                    epochs=20, batch_size=256, 
                    validation_split=0.1, 
                    verbose=1)

# Predict
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Debug
print(f"y_test shape: {y_test.shape}")
print(f"y_pred shape: {y_pred.shape}")
print(f"y_test range: {np.min(y_test)} to {np.max(y_test)}")
print(f"y_pred range: {np.min(y_pred)} to {np.max(y_pred)}")

# Check error in scaled space
mse_scaled = mean_squared_error(scaler_y.transform(y_test), y_pred_scaled)
print(f"Scaled RMSE: {np.sqrt(mse_scaled)}")

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Final Test RMSE: {rmse:.6f}")
