

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("datasets/gestures.csv", header=None)

# Split features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#  Improved model
model = RandomForestClassifier(
    n_estimators=300,       
    max_depth=15,            
    min_samples_split=5,    
    min_samples_leaf=2,
    random_state=42
)

# Train
model.fit(X_train, y_train)
# Predict
y_pred = model.predict(X_test)
# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\n Classification Report:")
print(classification_report(y_test, y_pred))
# Cross-validation 
cv_scores = cross_val_score(model, X, y, cv=5)
print("\n Cross-validation Accuracy:", cv_scores.mean())

# Save model
with open("models/gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved successfully!")