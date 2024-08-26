import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load and inspect data
crop = pd.read_csv("Crop_recommendation.csv")

# Clean and preprocess
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9,
    'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
    'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}
crop['label'] = crop['label'].map(crop_dict)

# Split features and target
X = crop.drop('label', axis=1)
Y = crop['label']

# Check class distribution
print("Class distribution:")
print(Y.value_counts())

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature Scaling
mx = MinMaxScaler()
sc = StandardScaler()

X_train = mx.fit_transform(X_train)
X_test = mx.transform(X_test)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Model Training
randclf = RandomForestClassifier(random_state=42)
randclf.fit(X_train, Y_train)

# Evaluate Model
y_pred = randclf.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print("RandomForest accuracy:", accuracy)

# Check the distribution of predicted labels
print("Predicted labels distribution:")
print(pd.Series(y_pred).value_counts())

# Recommendation function
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features = mx.transform(features)  # Only transform, do not fit
    features = sc.transform(features)  # Only transform, do not fit
    prediction = randclf.predict(features)
    return prediction[0]

# Test the recommendation function with an example input
N = 90
P = 42
K = 43
temperature = 20.88
humidity = 82.00
ph = 6.50
rainfall = 202.94

predicted_crop = recommendation(N, P, K, temperature, humidity, ph, rainfall)
print("Predicted crop:", predicted_crop)

# Save the model and scalers using pickle
pickle.dump(randclf, open('model.pkl', 'wb'))
pickle.dump(mx, open('MinMaxScaler.pkl', 'wb'))
pickle.dump(sc, open('StandardScaler.pkl', 'wb'))
