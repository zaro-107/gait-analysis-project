import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# Optional XGBoost
HAS_XGB = True
try:
    from xgboost import XGBClassifier
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Random Forest and Decision Tree will still run.")


# =========================
# 1. LOAD DATA
# =========================
PARQUET_PATH = r"C:\Users\ashok\OneDrive\Desktop\gait\gait_dataset.parquet"  

df = pd.read_parquet(PARQUET_PATH)

print("=" * 60)
print("DATASET SHAPE:", df.shape)
print("=" * 60)
print("COLUMNS:")
print(df.columns.tolist())
print("=" * 60)
print("FIRST 5 ROWS:")
print(df.head())
print("=" * 60)

# Sensor columns
sensor_cols = [str(i) for i in range(39) if str(i) in df.columns]

if len(sensor_cols) == 0:
    raise ValueError("No sensor columns found. Expected columns '0' to '38'.")

print(f"Found {len(sensor_cols)} sensor columns.")


# =========================
# 2. CREATE LABELS FROM source_file
# =========================
def extract_activity_label(path_text):
    text = str(path_text).lower()

    # common gait/activity keywords
    if "walk" in text:
        return "walking"
    elif "stand" in text:
        return "standing"
    elif "sit" in text:
        return "sitting"
    elif "stairsup" in text or "upstairs" in text or "stair_up" in text:
        return "stairs_up"
    elif "stairsdown" in text or "downstairs" in text or "stair_down" in text:
        return "stairs_down"
    elif "run" in text:
        return "running"
    elif "jump" in text:
        return "jumping"
    else:
        return "other"

if "source_file" not in df.columns:
    raise ValueError("Column 'source_file' not found in dataset.")

df["activity"] = df["source_file"].apply(extract_activity_label)

print("\nACTIVITY COUNTS:")
print(df["activity"].value_counts())
print("=" * 60)

# Remove "other" if too noisy
df = df[df["activity"] != "other"].copy()

if df.empty:
    raise ValueError("All rows became 'other'. Check source_file naming pattern first.")

print("AFTER FILTERING 'other':", df.shape)
print(df["activity"].value_counts())
print("=" * 60)


# =========================
# 3. BASIC FEATURE ENGINEERING
# =========================
features = pd.DataFrame(index=df.index)

# Row-level stats across all 39 sensor channels
features["mean"] = df[sensor_cols].mean(axis=1)
features["std"] = df[sensor_cols].std(axis=1)
features["min"] = df[sensor_cols].min(axis=1)
features["max"] = df[sensor_cols].max(axis=1)
features["range"] = features["max"] - features["min"]
features["median"] = df[sensor_cols].median(axis=1)
features["q1"] = df[sensor_cols].quantile(0.25, axis=1)
features["q3"] = df[sensor_cols].quantile(0.75, axis=1)
features["iqr"] = features["q3"] - features["q1"]
features["abs_mean"] = df[sensor_cols].abs().mean(axis=1)
features["energy"] = (df[sensor_cols] ** 2).mean(axis=1)
features["rms"] = np.sqrt((df[sensor_cols] ** 2).mean(axis=1))
features["zero_cross_approx"] = (np.diff(np.sign(df[sensor_cols].values), axis=1) != 0).sum(axis=1)

# Add metadata
if "subject_id" in df.columns:
    features["subject_id"] = df["subject_id"]

features["activity"] = df["activity"]

print("FEATURE DATASET SHAPE:", features.shape)
print(features.head())
print("=" * 60)


# =========================
# 4. SAVE FEATURE DATASET
# =========================
os.makedirs("outputs", exist_ok=True)
features.to_csv("outputs/engineered_gait_features.csv", index=False)
print("Saved: outputs/engineered_gait_features.csv")


# =========================
# 5. EDA GRAPHS
# =========================
plt.figure(figsize=(8, 5))
features["activity"].value_counts().plot(kind="bar")
plt.title("Activity Distribution")
plt.xlabel("Activity")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/activity_distribution.png")
plt.show()

numeric_cols = [
    "mean", "std", "min", "max", "range",
    "median", "q1", "q3", "iqr", "abs_mean",
    "energy", "rms", "zero_cross_approx"
]

for col in ["mean", "std", "range", "energy"]:
    plt.figure(figsize=(8, 5))
    for label in features["activity"].unique():
        subset = features[features["activity"] == label][col]
        plt.hist(subset, bins=30, alpha=0.5, label=label)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/{col}_distribution.png")
    plt.show()


# =========================
# 6. PREPARE DATA FOR MODELS
# =========================
X = features[numeric_cols].copy()
y = features["activity"].copy()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("TRAIN SHAPE:", X_train.shape)
print("TEST SHAPE :", X_test.shape)
print("CLASSES    :", list(label_encoder.classes_))
print("=" * 60)


# =========================
# 7. TRAIN MODELS
# =========================
results = []

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
results.append(("Random Forest", rf_acc))

print("\nRANDOM FOREST ACCURACY:", rf_acc)
print(classification_report(y_test, rf_pred, target_names=label_encoder.classes_))

# Decision Tree
dt = DecisionTreeClassifier(
    random_state=42,
    max_depth=10,
    class_weight="balanced"
)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
results.append(("Decision Tree", dt_acc))

print("\nDECISION TREE ACCURACY:", dt_acc)
print(classification_report(y_test, dt_pred, target_names=label_encoder.classes_))

# XGBoost
if HAS_XGB:
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        scale_pos_weight=5
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    results.append(("XGBoost", xgb_acc))

    print("\nXGBOOST ACCURACY:", xgb_acc)
    print(classification_report(y_test, xgb_pred, target_names=label_encoder.classes_))


# =========================
# 8. MODEL COMPARISON GRAPH
# =========================
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
print("\nMODEL COMPARISON:")
print(results_df)

results_df.to_csv("outputs/model_comparison.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("outputs/model_accuracy_comparison.png")
plt.show()


# =========================
# 9. FEATURE IMPORTANCE (RF)
# =========================
importances = pd.DataFrame({
    "Feature": numeric_cols,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nRANDOM FOREST FEATURE IMPORTANCE:")
print(importances)

importances.to_csv("outputs/rf_feature_importance.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(importances["Feature"], importances["Importance"])
plt.title("Random Forest Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/rf_feature_importance.png")
plt.show()


# =========================
# 10. CONFUSION MATRIX (RF)
# =========================
cm = confusion_matrix(y_test, rf_pred)

print("\nRANDOM FOREST CONFUSION MATRIX:")
print(cm)

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Random Forest Confusion Matrix")
plt.colorbar()
plt.xticks(np.arange(len(label_encoder.classes_)), label_encoder.classes_, rotation=45)
plt.yticks(np.arange(len(label_encoder.classes_)), label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/rf_confusion_matrix.png")
plt.show()

print("\nAll outputs saved in 'outputs/' folder.")
print("Done.")