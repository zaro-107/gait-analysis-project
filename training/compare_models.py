import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from feature_extraction import create_dataset


# =========================
# CONFIG
# =========================
DATA_PATH = r"C:\Users\ashok\OneDrive\Desktop\gait\gait_dataset.parquet"
SAVE_DIR = r"C:\Users\ashok\OneDrive\Desktop\gait\training\outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

WINDOW_SIZE = 128
STEP = 64
RANDOM_STATE = 42


# =========================
# LOAD DATA
# =========================
print("Loading data...")
df = pd.read_parquet(DATA_PATH)
print("Raw dataframe shape:", df.shape)

print("Creating windowed dataset...")
X, y, groups = create_dataset(df, window_size=WINDOW_SIZE, step=STEP)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("groups shape:", groups.shape)
print("Classes before cleaning:", np.unique(y))

# Optional: remove bad / ambiguous / zero-support class
mask = y != "down"
X = X[mask]
y = y[mask]
groups = groups[mask]

print("Classes after cleaning:", np.unique(y))
print("Filtered X shape:", X.shape)


# =========================
# ENCODE LABELS
# =========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Encoded classes:", list(le.classes_))


# =========================
# GROUP SPLIT
# =========================
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)

for train_idx, test_idx in gss.split(X, y_encoded, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)


# =========================
# MODELS
# =========================
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "DecisionTree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        random_state=RANDOM_STATE
    )
}


# =========================
# TRAIN + EVALUATE
# =========================
results = {}
trained_models = {}

for name, model in models.items():
    print(f"\n{'=' * 50}")
    print(f"Training {name}...")
    print(f"{'=' * 50}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    results[name] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }
    trained_models[name] = model

    print(f"{name} Accuracy     : {acc:.4f}")
    print(f"{name} Macro F1     : {macro_f1:.4f}")
    print(f"{name} Weighted F1  : {weighted_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
        zero_division=0
    ))

    # Save report to txt file
    report_path = os.path.join(SAVE_DIR, f"{name}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{name} Accuracy: {acc:.4f}\n")
        f.write(f"{name} Macro F1: {macro_f1:.4f}\n")
        f.write(f"{name} Weighted F1: {weighted_f1:.4f}\n\n")
        f.write(classification_report(
            y_test,
            y_pred,
            target_names=le.classes_,
            zero_division=0
        ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{name}_confusion_matrix.png"))
    plt.show()


# =========================
# MODEL COMPARISON PLOTS
# =========================
model_names = list(results.keys())
accuracies = [results[m]["accuracy"] for m in model_names]
macro_f1s = [results[m]["macro_f1"] for m in model_names]
weighted_f1s = [results[m]["weighted_f1"] for m in model_names]

# Accuracy plot
plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, accuracies)
plt.title("Model Comparison - Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

for bar, val in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
             ha="center", va="bottom")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "model_accuracy_comparison.png"))
plt.show()

# Macro F1 plot
plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, macro_f1s)
plt.title("Model Comparison - Macro F1")
plt.ylabel("Macro F1")
plt.ylim(0, 1)

for bar, val in zip(bars, macro_f1s):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
             ha="center", va="bottom")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "model_macro_f1_comparison.png"))
plt.show()

# Weighted F1 plot
plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, weighted_f1s)
plt.title("Model Comparison - Weighted F1")
plt.ylabel("Weighted F1")
plt.ylim(0, 1)

for bar, val in zip(bars, weighted_f1s):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
             ha="center", va="bottom")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "model_weighted_f1_comparison.png"))
plt.show()


# =========================
# BEST MODEL FEATURE IMPORTANCE
# =========================
best_model_name = max(results, key=lambda k: results[k]["accuracy"])
best_model = trained_models[best_model_name]

print(f"\nBest model: {best_model_name}")
print("Best model metrics:", results[best_model_name])

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_

    # Top 30 feature importances
    top_n = min(30, len(importances))
    top_idx = np.argsort(importances)[::-1][:top_n]
    top_vals = importances[top_idx]

    plt.figure(figsize=(10, 6))
    plt.bar(range(top_n), top_vals)
    plt.xticks(range(top_n), top_idx, rotation=90)
    plt.title(f"Top {top_n} Feature Importances - {best_model_name}")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{best_model_name}_feature_importance.png"))
    plt.show()


# =========================
# SAVE RESULT SUMMARY CSV
# =========================
results_df = pd.DataFrame(results).T
results_df.to_csv(os.path.join(SAVE_DIR, "model_results_summary.csv"), index=True)

print("\nAll done.")
print("Files saved in:", SAVE_DIR)