import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error

# Load and Reshape data to 2D
data = scipy.io.loadmat('/home/hzy9981/datax/dataX.mat')
X = data['inputtrain'].astype('float32')
y = data['outputtrain'].astype('float32')

# Scaling (using global min/max or std, here simple global scaling for 2D)
X_mean, X_std = X.mean(), X.std()
y_mean, y_std = y.mean(), y.std()

X_scaled = (X - X_mean) / X_std
y_scaled = (y - y_mean) / y_std

X_train = X_scaled.reshape(1, 500, 500, 1)
y_train = y_scaled.reshape(1, 500, 500, 1)

# A simple U-Net-like or CNN architecture to learn spatial mapping
model = keras.Sequential([
    layers.Input(shape=(500, 500, 1)),
    layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.Conv2D(1, kernel_size=(3, 3), padding='same', activation='linear')
])

model.compile(optimizer='adam', loss='mse')

# Train
# Increased epochs
history = model.fit(X_train, y_train, epochs=500, verbose=1)

# Predict
y_pred_scaled = model.predict(X_train)
y_pred = y_pred_scaled * y_std + y_mean

# Evaluate (using original scale)
rmse = np.sqrt(mean_squared_error(y.flatten(), y_pred.flatten()))
print(f"Final 2D CNN RMSE: {rmse:.6f}")

if rmse < 0.01:
    print("SUCCESS: RMSE is less than 0.01")
    model.save('model_cnn.keras')
else:
    print("FAILURE: RMSE is still too high")
