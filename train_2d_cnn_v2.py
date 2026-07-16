import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error

# Load data
data = scipy.io.loadmat('dataX.mat')
X_raw = data['inputtrain'].astype('float32').reshape(500, 500)
y_raw = data['outputtrain'].astype('float32').reshape(500, 500)

# Target: residual
target = y_raw - (X_raw - 1.0)

# Scale Input
X_mean, X_std = X_raw.mean(), X_raw.std()
X_scaled = (X_raw - X_mean) / X_std

# Reshape for CNN
X_input = X_scaled.reshape(1, 500, 500, 1)
y_target = target.reshape(1, 500, 500, 1)

# Architecture: A deeper CNN to capture complex spatial patterns
model = keras.Sequential([
    layers.Input(shape=(500, 500, 1)),
    layers.Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu'),
    layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.Conv2D(1, kernel_size=(3, 3), padding='same', activation='linear')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Train
# We'll use more epochs but small verbose
print("Starting training on residual...")
for epoch in range(1, 101):
    history = model.fit(X_input, y_target, epochs=1, verbose=0)
    loss = history.history['loss'][0]
    if epoch % 10 == 0:
        y_pred_res = model.predict(X_input, verbose=0)
        rmse = np.sqrt(np.mean((y_target - y_pred_res)**2))
        print(f"Epoch {epoch}, Loss: {loss:.6f}, RMSE: {rmse:.6f}")
        if rmse < 0.01:
            print("SUCCESS: RMSE is less than 0.01")
            break

# Final Evaluation on full data (including test if we want, but let's focus on train first)
y_pred_res = model.predict(X_input, verbose=0)
y_pred = y_pred_res.flatten() + (X_raw.flatten() - 1.0)
final_rmse = np.sqrt(mean_squared_error(y_raw.flatten(), y_pred))
print(f"Final Train RMSE: {final_rmse:.6f}")

if final_rmse < 0.01:
    model.save('model_cnn_residual.keras')
    print("Model saved.")
