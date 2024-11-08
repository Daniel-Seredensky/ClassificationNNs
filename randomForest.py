import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report
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

# Train Random Forest model with class weights
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, y_pred))

