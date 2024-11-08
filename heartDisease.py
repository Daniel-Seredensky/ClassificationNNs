import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import numpy as np

# Load data
features = pd.read_csv('HeartDisease/heart_disease_features.csv')
targets = pd.read_csv('HeartDisease/heart_disease_target.csv')

# Clean data - drop rows with NaN values
features = features.dropna()
targets = targets.loc[features.index]

# Combine features and targets for resampling
data = pd.concat([features, targets], axis=1)

# Separate majority and minority classes
majority_class = data[data.iloc[:, -1] == 0]
minority_classes = data[data.iloc[:, -1] != 0]

# Upsample minority classes
minority_upsampled = resample(minority_classes,
                              replace=True,  # sample with replacement
                              n_samples=len(majority_class),  # match number of majority class
                              random_state=42)

# Combine majority class with upsampled minority classes
data_upsampled = pd.concat([majority_class, minority_upsampled])

# Split features and targets
features = data_upsampled.iloc[:, :-1]
targets = data_upsampled.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode targets
y_train = tf.keras.utils.to_categorical(y_train, num_classes=5)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=5)

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
eval_result = model.evaluate(X_test, y_test)
print(f"Test Loss: {eval_result[0]}, Test Accuracy: {eval_result[1]}")