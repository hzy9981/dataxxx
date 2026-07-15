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

# Scale Input ONLY
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Build model
model = keras.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# Use a smaller learning rate and direct MSE optimization
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Train model
# Increase epochs for better convergence
history = model.fit(X_train_scaled, y_train, 
                    epochs=100, batch_size=128, 
                    validation_split=0.1, 
                    verbose=1)

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
