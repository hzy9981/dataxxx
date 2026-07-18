import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error

# 1. Load data
data = scipy.io.loadmat('dataX.mat')
X_raw_full = data['inputtrain'].astype('float32').reshape(500, 500)
y_raw_full = data['outputtrain'].astype('float32').reshape(500, 500)

X_test_raw_full = data['inputtest'].astype('float32').reshape(500, 500)
y_test_raw_full = data['outputtest'].astype('float32').reshape(500, 500)

# Target: residual
target_full = y_raw_full - (X_raw_full - 1.0)
target_test_full = y_test_raw_full - (X_test_raw_full - 1.0)

# Scaling using training stats
X_mean, X_std = X_raw_full.mean(), X_raw_full.std()
X_scaled_full = (X_raw_full - X_mean) / X_std
X_test_scaled_full = (X_test_raw_full - X_mean) / X_std

# 2. Patch Extraction Function
def extract_patches(X, y, patch_size=64, stride=32):
    patches_X = []
    patches_y = []
    h, w = X.shape
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patches_X.append(X[i:i+patch_size, j:j+patch_size])
            patches_y.append(y[i:i+patch_size, j:j+patch_size])
    return np.array(patches_X)[..., np.newaxis], np.array(patches_y)[..., np.newaxis]

PATCH_SIZE = 64
STRIDE = 16 # Overlapping patches to increase data
X_train_patches, y_train_patches = extract_patches(X_scaled_full, target_full, PATCH_SIZE, STRIDE)
print(f"Extracted {len(X_train_patches)} training patches.")

# 3. Data Augmentation Pipeline
def augment(x, y):
    # Random Flip
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_up_down(y)
    # Random Rotation (90, 180, 270)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    x = tf.image.rot90(x, k)
    y = tf.image.rot90(y, k)
    return x, y

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_patches, y_train_patches))
train_dataset = train_dataset.shuffle(1000).map(augment).batch(32).prefetch(tf.data.AUTOTUNE)

# 4. Model Architecture (Deeper with Skip Connections)
# Using (None, None, 1) to allow the model to handle both 64x64 patches and 500x500 full images
inputs = layers.Input(shape=(None, None, 1))
x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
x = layers.BatchNormalization()(x)

# Residual Block 1
res = x
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3, padding='same')(x)
x = layers.Add()([res, x])
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)

# Residual Block 2
res = x
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3, padding='same')(x)
x = layers.Add()([res, x])
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)

outputs = layers.Conv2D(1, 3, padding='same')(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='mse')

# 5. Training
print("Starting patch-based training with augmentation...")
callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, min_lr=1e-6, verbose=1),
    keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
]

model.fit(train_dataset, epochs=5000, callbacks=callbacks)

# 6. Evaluation on Full Image (Inference)
def predict_full_image(model, full_image_scaled):
    # Full Image Inference
    h, w = full_image_scaled.shape
    input_tensor = full_image_scaled.reshape(1, h, w, 1)
    prediction = model.predict(input_tensor, verbose=0)
    # Result will be (1, H, W, 1), squeeze to (H, W)
    return np.squeeze(prediction)

print("\nFinal Evaluation...")
y_pred_res_train = predict_full_image(model, X_scaled_full)
y_pred_train = y_pred_res_train.flatten() + (X_raw_full.flatten() - 1.0)
train_rmse = np.sqrt(mean_squared_error(y_raw_full.flatten(), y_pred_train))
print(f"Final Train RMSE: {train_rmse:.6f}")

y_pred_res_test = predict_full_image(model, X_test_scaled_full)
y_pred_test = y_pred_res_test.flatten() + (X_test_raw_full.flatten() - 1.0)
test_rmse = np.sqrt(mean_squared_error(y_test_raw_full.flatten(), y_pred_test))
print(f"Final Test RMSE: {test_rmse:.6f}")

if train_rmse < 0.06:
    model.save('model_patch_cnn.keras')
    print("Model saved.")
