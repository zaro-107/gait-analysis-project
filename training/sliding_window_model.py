import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

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
PARQUET_PATH = r"C:\Users\ashok\OneDrive\Desktop\gait\gait_dataset.parquet"   # change if needed

df = pd.read_parquet(PARQUET_PATH)

sensor_cols = [str(i) for i in range(39) if str(i) in df.columns]
if len(sensor_cols) == 0:
    raise ValueError("No sensor columns found. Expected columns '0' to '38'.")

if "source_file" not in df.columns:
    raise ValueError("Column 'source_file' not found.")

if "subject_id" not in df.columns:
    raise ValueError("Column 'subject_id' not found.")

print("=" * 60)
print("Original shape:", df.shape)
print("=" * 60)


# =========================
# 2. LABEL EXTRACTION
# =========================
def extract_activity_label(path_text):
    text = str(path_text).lower()

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

df["activity"] = df["source_file"].apply(extract_activity_label)

print("Activity counts before filtering:")
print(df["activity"].value_counts())
print("=" * 60)

df = df[df["activity"].isin(["walking", "standing", "sitting", "running"])].copy()

print("Activity counts after filtering:")
print(df["activity"].value_counts())
print("=" * 60)


# =========================
# 3. SLIDING WINDOW SETTINGS
# =========================
WINDOW_SIZE = 128
STEP_SIZE = 64

print(f"WINDOW_SIZE = {WINDOW_SIZE}")
print(f"STEP_SIZE   = {STEP_SIZE}")
print("=" * 60)


# =========================
# 4. WINDOW FEATURE FUNCTION
# =========================
def extract_window_features(window_df, sensor_cols):
    Xw = window_df[sensor_cols].values.astype(np.float32)

    feat = {}

    # Global statistics over the full window matrix
    feat["global_mean"] = Xw.mean()
    feat["global_std"] = Xw.std()
    feat["global_min"] = Xw.min()
    feat["global_max"] = Xw.max()
    feat["global_range"] = feat["global_max"] - feat["global_min"]
    feat["global_median"] = np.median(Xw)
    feat["global_abs_mean"] = np.abs(Xw).mean()
    feat["global_energy"] = np.mean(Xw ** 2)
    feat["global_rms"] = np.sqrt(np.mean(Xw ** 2))

    # Per-channel summaries, then average them
    ch_mean = Xw.mean(axis=0)
    ch_std = Xw.std(axis=0)
    ch_min = Xw.min(axis=0)
    ch_max = Xw.max(axis=0)
    ch_range = ch_max - ch_min
    ch_energy = np.mean(Xw ** 2, axis=0)
    ch_abs_mean = np.mean(np.abs(Xw), axis=0)

    feat["mean_of_channel_means"] = ch_mean.mean()
    feat["std_of_channel_means"] = ch_mean.std()
    feat["mean_of_channel_stds"] = ch_std.mean()
    feat["std_of_channel_stds"] = ch_std.std()
    feat["mean_of_channel_ranges"] = ch_range.mean()
    feat["mean_of_channel_energy"] = ch_energy.mean()
    feat["mean_of_channel_abs_mean"] = ch_abs_mean.mean()

    # Temporal changes
    diff = np.diff(Xw, axis=0)
    feat["mean_abs_diff"] = np.mean(np.abs(diff))
    feat["std_abs_diff"] = np.std(np.abs(diff))
    feat["diff_energy"] = np.mean(diff ** 2)

    # Zero crossing approx over time per channel
    sign_changes = np.diff(np.sign(Xw), axis=0) != 0
    feat["mean_zero_crossings"] = sign_changes.sum(axis=0).mean()

    return feat


# =========================
# 5. BUILD WINDOW DATASET
# =========================
window_rows = []

group_cols = ["subject_id", "source_file", "activity"]

for (subject_id, source_file, activity), group in df.groupby(group_cols):
    group = group.reset_index(drop=True)

    if len(group) < WINDOW_SIZE:
        continue

    for start in range(0, len(group) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window = group.iloc[start:end]

        feats = extract_window_features(window, sensor_cols)
        feats["subject_id"] = subject_id
        feats["source_file"] = source_file
        feats["activity"] = activity
        feats["start_idx"] = start
        feats["end_idx"] = end

        window_rows.append(feats)

window_df = pd.DataFrame(window_rows)

if window_df.empty:
    raise ValueError("No sliding windows were created. Try reducing WINDOW_SIZE.")

print("Window dataset shape:", window_df.shape)
print(window_df.head())
print("=" * 60)

print("Window activity counts:")
print(window_df["activity"].value_counts())
print("=" * 60)


# =========================
# 6. SAVE WINDOW FEATURES
# =========================
os.makedirs("outputs", exist_ok=True)
window_df.to_csv("outputs/sliding_window_features.csv", index=False)
print("Saved: outputs/sliding_window_features.csv")


# =========================
# 7. EDA
# =========================
plt.figure(figsize=(8, 5))
window_df["activity"].value_counts().plot(kind="bar")
plt.title("Sliding Window Activity Distribution")
plt.xlabel("Activity")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/sliding_window_activity_distribution.png")
plt.show()

plot_cols = ["global_mean", "global_std", "global_range", "global_energy", "mean_abs_diff"]
for col in plot_cols:
    plt.figure(figsize=(8, 5))
    for label in window_df["activity"].unique():
        subset = window_df[window_df["activity"] == label][col]
        plt.hist(subset, bins=30, alpha=0.5, label=label)
    plt.title(f"{col} Distribution by Activity")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/{col}_window_distribution.png")
    plt.show()


# =========================
# 8. PREPARE TRAIN / TEST
# =========================
drop_cols = ["subject_id", "source_file", "activity", "start_idx", "end_idx"]
feature_cols = [c for c in window_df.columns if c not in drop_cols]

X = window_df[feature_cols].copy()
y = window_df["activity"].copy()

le = LabelEncoder()
y_encoded = le.fit_transform(y)

from sklearn.model_selection import GroupShuffleSplit

groups = window_df["subject_id"]

gss = GroupShuffleSplit(test_size=0.2, random_state=42)

train_idx, test_idx = next(gss.split(X, y_encoded, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)
print("Classes    :", list(le.classes_))
print("=" * 60)


# =========================
# 9. TRAIN MODELS
# =========================
results = []

rf = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
results.append(("Random Forest", rf_acc))

print("\nRANDOM FOREST ACCURACY:", rf_acc)
print(classification_report(y_test, rf_pred, target_names=le.classes_))

dt = DecisionTreeClassifier(
    random_state=42,
    max_depth=12,
    class_weight="balanced"
)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
results.append(("Decision Tree", dt_acc))

print("\nDECISION TREE ACCURACY:", dt_acc)
print(classification_report(y_test, dt_pred, target_names=le.classes_))

if HAS_XGB:
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    results.append(("XGBoost", xgb_acc))

    print("\nXGBOOST ACCURACY:", xgb_acc)
    print(classification_report(y_test, xgb_pred, target_names=le.classes_))


# =========================
# 10. MODEL COMPARISON
# =========================
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
print("\nMODEL COMPARISON:")
print(results_df)

results_df.to_csv("outputs/sliding_window_model_comparison.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.title("Sliding Window Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("outputs/sliding_window_model_accuracy.png")
plt.show()


# =========================
# 11. RF FEATURE IMPORTANCE
# =========================
importances = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nRANDOM FOREST FEATURE IMPORTANCE:")
print(importances)

importances.to_csv("outputs/sliding_window_rf_feature_importance.csv", index=False)

plt.figure(figsize=(10, 6))
plt.bar(importances["Feature"], importances["Importance"])
plt.title("Sliding Window RF Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig("outputs/sliding_window_rf_feature_importance.png")
plt.show()


# =========================
# 12. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, rf_pred)

print("\nRANDOM FOREST CONFUSION MATRIX:")
print(cm)

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Sliding Window RF Confusion Matrix")
plt.colorbar()
plt.xticks(np.arange(len(le.classes_)), le.classes_, rotation=45)
plt.yticks(np.arange(len(le.classes_)), le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/sliding_window_rf_confusion_matrix.png")
plt.show()

print("\nAll outputs saved in outputs/")
print("Done.")