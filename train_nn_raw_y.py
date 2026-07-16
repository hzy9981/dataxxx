import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
data = scipy.io.loadmat('/home/hzy9981/datax/dataX.mat')
X_raw = data['inputtrain'].flatten()
y_train = data['outputtrain'].flatten()
X_test_raw = data['inputtest'].flatten()
y_test = data['outputtest'].flatten()

def create_features(x):
    # Lags and Rolling stats
    features = [x]
    for lag in [1, 2, 3, 5]:
        shifted = np.roll(x, lag)
        shifted[:lag] = x[0]
        features.append(shifted)
    
    # Rolling mean/std
    for window in [3, 5, 10]:
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        moving_avg = (cumsum[window:] - cumsum[:-window]) / window
        moving_avg = np.pad(moving_avg, (window-1, 0), 'constant', constant_values=x[0])
        features.append(moving_avg)

    return np.column_stack(features)

X_train = create_features(X_raw)
X_test = create_features(X_test_raw)

# Scale Input ONLY
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Build model
model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])

# ... rest of the training code ...

# Use a learning rate scheduler
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss='mse')

# Train model - Increased epochs with EarlyStopping
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train, 
                    epochs=300, batch_size=64, 
                    validation_split=0.1, 
                    callbacks=[callback], verbose=1)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Final Test RMSE: {rmse:.6f}")

if rmse < 0.01:
    print("SUCCESS: RMSE is less than 0.01")
    model.save('model.keras')
else:
    print("FAILURE: RMSE is still too high")
