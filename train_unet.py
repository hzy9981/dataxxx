import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error

# Load data
data = scipy.io.loadmat('dataX.mat')
X_train_raw = data['inputtrain'].astype('float32').reshape(500, 500)
y_train_raw = data['outputtrain'].astype('float32').reshape(500, 500)
X_test_raw = data['inputtest'].astype('float32').reshape(500, 500)
y_test_raw = data['outputtest'].astype('float32').reshape(500, 500)

# Residual learning
y_train_target = y_train_raw - (X_train_raw - 1.0)
y_test_target = y_test_raw - (X_test_raw - 1.0)

# Scale
X_mean, X_std = X_train_raw.mean(), X_train_raw.std()
X_train_scaled = (X_train_raw - X_mean) / X_std
X_test_scaled = (X_test_raw - X_mean) / X_std

X_train_in = X_train_scaled.reshape(1, 500, 500, 1)
X_test_in = X_test_scaled.reshape(1, 500, 500, 1)
y_train_out = y_train_target.reshape(1, 500, 500, 1)
y_test_out = y_test_target.reshape(1, 500, 500, 1)

def build_unet(input_shape):
    inputs = layers.Input(input_shape)
    
    # Simple U-Net with padding to keep 500x500
    # Note: 500 is not a power of 2, so we need to be careful with pooling/up-sampling
    # or just use padding/cropping.
    # Alternatively, just use a deep ResNet-like CNN.
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    
    # Skip connections
    c1 = x
    
    x = layers.MaxPooling2D(2)(x) # 250x250
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    
    c2 = x
    
    x = layers.MaxPooling2D(2)(x) # 125x125
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    
    x = layers.UpSampling2D(2)(x) # 250x250
    x = layers.Concatenate()([x, c2])
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    
    x = layers.UpSampling2D(2)(x) # 500x500
    x = layers.Concatenate()([x, c1])
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    
    outputs = layers.Conv2D(1, 1, padding='same', activation='linear')(x)
    
    return keras.Model(inputs, outputs)

model = build_unet((500, 500, 1))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

print("Training U-Net on residual...")
for epoch in range(1, 101):
    model.fit(X_train_in, y_train_out, epochs=1, verbose=0)
    if epoch % 10 == 0:
        y_pred = model.predict(X_test_in, verbose=0)
        rmse = np.sqrt(np.mean((y_test_out - y_pred)**2))
        print(f"Epoch {epoch}, Test RMSE: {rmse:.6f}")
        if rmse < 0.01:
            print("SUCCESS")
            break

# Final
y_pred = model.predict(X_test_in, verbose=0)
final_rmse = np.sqrt(np.mean((y_test_out - y_pred)**2))
print(f"Final Test RMSE: {final_rmse:.6f}")
if final_rmse < 0.01:
    model.save('model_unet.keras')
