# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model

# Load the dataset
file_path = 'cleaned_world_air_quality (1).csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Data Preprocessing
# Select relevant features and target
features = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value', 'lat', 'lng']
target = 'AQI Category'

# Encode the target variable
label_encoder = LabelEncoder()
data[target] = label_encoder.fit_transform(data[target])

# Normalize numerical features using Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data[features])

# Reshape data for CNN (into 2D grid with 1 channel)
X_reshaped = X_scaled.reshape(-1, 2, 3, 1)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, data[target], test_size=0.2, random_state=42)

# Build the CNN Model
model = Sequential([
    Conv2D(32, (2, 2), activation='relu', input_shape=(2, 3, 1)),
    MaxPooling2D((1, 1)),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer
])

# Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the Model on Test Set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the Model
model.save('air_quality_cnn_model.h5')
print("Model saved as 'air_quality_cnn_model.h5'")

# Load the Model for Predictions (Optional)
model = load_model('air_quality_cnn_model.h5')
predictions = model.predict(X_test)

