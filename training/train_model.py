import pandas as pd
import joblib
import os   #  ADD THIS

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from feature_extraction import create_dataset

# Load data
df = pd.read_parquet(r"C:\Users\ashok\OneDrive\Desktop\gait\gait_dataset.parquet")

# Create dataset
X, y = create_dataset(df)

print("Shape:", X.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))



# Root directory (gait folder)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# saved_models folder
MODEL_DIR = os.path.join(ROOT_DIR, "saved_models")

# Create folder if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Final model path
model_path = os.path.join(MODEL_DIR, "activity_model.pkl")

# Save model
joblib.dump(model, model_path)

print("Model saved at:", model_path)