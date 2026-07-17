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

X_test_raw = data['inputtest'].astype('float32').reshape(500, 500)
y_test_raw = data['outputtest'].astype('float32').reshape(500, 500)

# Target: residual
target = y_raw - (X_raw - 1.0)
target_test = y_test_raw - (X_test_raw - 1.0)

# Scale Input (Using Training Stats for both to ensure independence)
X_mean, X_std = X_raw.mean(), X_raw.std()
X_scaled = (X_raw - X_mean) / X_std
X_test_scaled = (X_test_raw - X_mean) / X_std

# Reshape for CNN
X_input = X_scaled.reshape(1, 500, 500, 1)
y_target = target.reshape(1, 500, 500, 1)
X_test_input = X_test_scaled.reshape(1, 500, 500, 1)
y_test_target = target_test.reshape(1, 500, 500, 1)

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

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mse')

# Train
# We'll use more epochs but small verbose
print("Starting training on residual...")
for epoch in range(1, 501):
    history = model.fit(X_input, y_target, epochs=1, verbose=0)
    loss = history.history['loss'][0]
    if epoch % 10 == 0:
        y_pred_res = model.predict(X_input, verbose=0)
        rmse = np.sqrt(np.mean((y_target - y_pred_res)**2))
        print(f"Epoch {epoch}, Loss: {loss:.6f}, RMSE: {rmse:.6f}")
        if rmse < 0.06:
            print("SUCCESS: RMSE is less than 0.06")
            break

# Final Evaluation
print("\nPerforming final evaluation...")
y_pred_res_train = model.predict(X_input, verbose=0)
y_pred_train = y_pred_res_train.flatten() + (X_raw.flatten() - 1.0)
train_rmse = np.sqrt(mean_squared_error(y_raw.flatten(), y_pred_train))
print(f"Final Train RMSE: {train_rmse:.6f}")

y_pred_res_test = model.predict(X_test_input, verbose=0)
y_pred_test = y_pred_res_test.flatten() + (X_test_raw.flatten() - 1.0)
test_rmse = np.sqrt(mean_squared_error(y_test_raw.flatten(), y_pred_test))
print(f"Final Test RMSE: {test_rmse:.6f}")

if train_rmse < 0.06:
    model.save('model_cnn_residual.keras')
    print("Model saved.")
