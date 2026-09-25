"""
Edited by Sangmork Park at VMI 
Last update: Sep. 2026

The current tensorflow does not support up-to-date python (3.13 ~ ).
We will use tehsorflow-nightly and keras-nightly version instead of stable versions.

$ pip install tf-nightly keras-nightly
$ pip install -U scikit-learn               // -U: update if necessary

"""

# Disable CUDA GPU and use CPU
# NVIDIA GeForceRTX5070GPU (Compute Capability 2.0) is too new for the CUDA compilation in tf-nightly
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf

""" keras-nightly version was installed """
# from tensorflow import keras
# from tensorflow.keras import layers, callbacks
# import tf_keras as keras
# from tf_keras import layers, callbacks
import keras
from keras import layers, callbacks

import pandas as pd


# -------------------------------------------------------------------------
# step-1. Data Preparation: simulation data generation 
# -------------------------------------------------------------------------
# 1-1. Generate 1,000 samples (10 classes * 100 samples) with 12 features
#   X: input feature matrix (2D numpy array of 1000 x 12), 
#   y: target label vector (1D numpy array of 1000 x 1)
"""**"""
# X, y = make_classification(
#     n_samples=1000,         # total number of rows
#     n_features=12,          # total number of columns (variables)
#     n_informative=10,       # 10 features out of 12 contain uactual useful information
#     n_redundant=2,          # remaining 2 features are useless 
#     n_classes=10,           # number of classes
#     n_clusters_per_class=1, # 10 classes will be grouped into 1 tight cluster in the data space
#     random_state=42         # seed number of random number generator
# )

# -------------------------------------------------------------------------
# step-1. Data Preparation
# -------------------------------------------------------------------------
# 1-1. Read data from a *.csv file
df = pd.read_csv("gesture_data.csv")        # read_csv() removes the head line
X_df = df.drop(columns='label')
y = df['label'].to_numpy() - 1

# Round all feature columns to 4 decimal places and create X
X_df = X_df.round(4)
X = X_df.to_numpy(dtype=np.float32)

# 1-2. Split data into 2 piecies: (70% Train, 15% Validation) + 15% Test (stratified to ensure balance)
#   stratify=y: forces the split to maintain the exact same calss proportion in both sets
#   X_train_val: 850 x 12, y_train_val, 850 x 1, X_test: 150 x 12, y_test: 150 x 1 
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# 1.3. Divide the train_val data set into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
)  # 0.1765 of 0.85 is ~0.15 of total

# 1.4. Standardize (normalize) features based on training distribution
#   normalize the data to ensure that all 12 features scaled a mean of 0 and avariance of 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)     # X_scaled = (X - mean) / variance
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# -------------------------------------------------------------------------
# Setp-2. Model Architecture
# -------------------------------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(12,)),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    # 10 classes with softmax output
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------------------------------------------------
# Step-3. Training & Validation Performance
# -------------------------------------------------------------------------
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-5,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=120,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

# -------------------------------------------------------------------------
# Setp-4. Evaluation on Test Set
# -------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%\n")

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------------------------------------------------
# Setp-5. Performance visualizations by plots
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Loss curves
axes[0].plot(history.history['loss'], label='Train Loss', color='steelblue', lw=2)
axes[0].plot(history.history['val_loss'], label='Val Loss', color='orange', lw=2, linestyle='--')
axes[0].set_title('Cross-Entropy Loss Improvement')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

# Plot 2: Accuracy curves
axes[1].plot(history.history['accuracy'], label='Train Accuracy', color='steelblue', lw=2)
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', color='orange', lw=2, linestyle='--')
axes[1].set_title('Accuracy Improvement')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend()

# Plot 3: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"C{i}" for i in range(10)])
disp.plot(ax=axes[2], cmap='Blues', colorbar=False)
axes[2].set_title(f'Test Confusion Matrix (Acc: {test_acc*100:.1f}%)')

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------
# Setp-6. Save the visualizations plots
# -------------------------------------------------------------------------

# 1. Save Loss Plot
plt.figure(figsize=(6, 5))
plt.plot(history.history['loss'], label='Train Loss', color='steelblue', lw=2)
plt.plot(history.history['val_loss'], label='Val Loss', color='orange', lw=2, linestyle='--')
plt.title('Cross-Entropy Loss Improvement')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('training_validation_loss.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Save Accuracy Plot
plt.figure(figsize=(6, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='steelblue', lw=2)
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange', lw=2, linestyle='--')
plt.title('Accuracy Improvement')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('training_validation_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Save Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"C{i}" for i in range(10)])
disp.plot(cmap='Blues', colorbar=True)
plt.title(f'Test Confusion Matrix (Acc: {test_acc*100:.1f}%)')
plt.tight_layout()
plt.savefig('test_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()