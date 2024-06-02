# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten

# Load the dataset
data = pd.read_csv('fer2013.csv')

# Inspect the dataset
print(data.head())
print(data.columns)

# Update these column names based on the fer2013 dataset
context_column = 'Usage'  # Context based on data usage (Train, PublicTest, PrivateTest)
target_column = 'emotion'  # Target is the emotion label

# Verify the column names
expected_columns = [context_column, 'pixels', target_column]
for col in expected_columns:
    if col not in data.columns:
        raise ValueError(f"Expected column '{col}' not found in the dataset")

# Convert pixels column to numpy arrays
data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))

# Separate features and target
X = np.stack(data['pixels'].values)
X = X / 255.0  # Normalize pixel values
y = data[target_column]

# Reshape X to (number of samples, 48, 48, 1) since images are 48x48 pixels
X = X.reshape(-1, 48, 48, 1)

# One-hot encode the target variable
y = pd.get_dummies(y).values

# One-hot encode the context variable
context_encoder = OneHotEncoder(sparse=False)
context_encoded = context_encoder.fit_transform(data[[context_column]])

# Flatten the image data
X_flat = X.reshape(X.shape[0], -1)

# Combine the flattened image data with the context information
X_combined = np.concatenate([X_flat, context_encoded], axis=1)

# Split the encoded data
X_train_enc, X_test_enc, y_train_enc, y_test_enc = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Neural network model
model = Sequential([
    Input(shape=(X_train_enc.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(y_train_enc.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_enc, y_train_enc, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the multi-context model
loss, accuracy = model.evaluate(X_test_enc, y_test_enc)
print(f'Multi-context model accuracy: {accuracy}')