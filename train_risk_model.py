# train_risk_model.py
# Is file ka kaam gait.csv se data padhna aur ek risk prediction model train karna hai.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Step 1: Loading and Preparing Data...")
# Dataset ko load karein
df = pd.read_csv('data/gait.csv')

# Features (X) aur Target (y) ko alag karein
# Hum 'angle' ko feature aur 'condition' ko target banayenge
X = df[['angle', 'time', 'joint']]
y = df['condition']

# Target ko asaan banayein: 1 (unbraced) ko 0 (Low Risk) aur 2,3 (braced) ko 1 (High Risk)
y = y.apply(lambda x: 0 if x == 1 else 1)

# Data ko training aur testing sets mein baantein
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data preparation complete.")

print("\nStep 2: Training the Risk Prediction Model...")
# Hum RandomForestClassifier istemal karenge, jo ek powerful model hai
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Model ko training data par train karein
model.fit(X_train, y_train)
print("Model training complete.")

print("\nStep 3: Evaluating the Model...")
# Model ki performance test karein
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nStep 4: Saving the Model...")
# Trained model ko ek file mein save karein taake hum isay app mein istemal kar sakein
joblib.dump(model, 'risk_model.pkl')
print("Model saved as 'risk_model.pkl'. You can now run the Streamlit app.")
