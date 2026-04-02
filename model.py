import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle

# Sample dataset (you can replace with real dataset)
data = {
    "followers": [100, 50, 300, 20, 500, 10],
    "following": [150, 2000, 180, 3000, 200, 5000],
    "posts": [10, 2, 50, 1, 100, 0],
    "bio_length": [20, 5, 50, 2, 80, 1],
    "profile_pic": [1, 0, 1, 0, 1, 0],
    "label": [0, 1, 0, 1, 0, 1]  # 0=Real, 1=Fake
}

df = pd.DataFrame(data)

# Features & Labels
X = df.drop("label", axis=1)
y = df["label"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train ANN model
model = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=500)
model.fit(X_scaled, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model and scaler saved successfully!")