# Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
file_path = 'cleaned_world_air_quality (1).csv'  # Update with the correct file path
data = pd.read_csv(file_path)

# Select features and target
features = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value', 'lat', 'lng']
target = 'AQI Category'

# Encode target variable
label_encoder = LabelEncoder()
data[target] = label_encoder.fit_transform(data[target])

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data[features])  # Ensure X_scaled is defined here
y = data[target]

# Define KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Perform k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(knn, X_scaled, y, cv=kfold, scoring='accuracy')

# Print cross-validation results
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Accuracy: {cv_scores.mean():.2f}")
print(f"Standard Deviation: {cv_scores.std():.2f}")
