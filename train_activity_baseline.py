import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier


# =========================
# CONFIG
# =========================
PARQUET_PATH = "hugodab_dataset.parquet"

# ✅ 3-class subject-independent (valid for HuGaDB in your folder)
KEEP_ACTIVITIES = ["walking", "sitting", "standing"]

WINDOW_SIZE = 256
STEP_SIZE = 128

N_SPLITS = 5
USE_FOLD_INDEX = 0


# =========================
# HELPERS
# =========================
def pick_sensor_columns(df: pd.DataFrame):
    meta = {"activity", "subject_id", "trial_id", "source_file"}
    return [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]


def fft_features_1d(x: np.ndarray):
    x = np.nan_to_num(x, nan=0.0)
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x))
    dom_bin = int(np.argmax(spec[1:]) + 1) if spec.shape[0] > 1 else 0
    spec_energy = float(np.mean(spec**2))
    return dom_bin, spec_energy


def window_features(Xw: np.ndarray) -> np.ndarray:
    Xw = np.nan_to_num(Xw, nan=0.0)

    mean = Xw.mean(axis=0)
    std = Xw.std(axis=0)
    mn = Xw.min(axis=0)
    mx = Xw.max(axis=0)
    energy_time = np.mean(Xw**2, axis=0)

    dom_bins = np.empty(Xw.shape[1], dtype=np.float32)
    spec_energy = np.empty(Xw.shape[1], dtype=np.float32)
    for ch in range(Xw.shape[1]):
        dom, en = fft_features_1d(Xw[:, ch])
        dom_bins[ch] = dom
        spec_energy[ch] = en

    return np.concatenate([mean, std, mn, mx, energy_time, dom_bins, spec_energy]).astype(np.float32)


def build_windows(df: pd.DataFrame, sensor_cols):
    X_list, y_list, group_list = [], [], []

    # important: never mix rows from different recordings
    for _, g in df.groupby("source_file", sort=False):
        activity = g["activity"].iloc[0]
        subject = g["subject_id"].iloc[0]

        X = g[sensor_cols].to_numpy(dtype=np.float32)
        n = X.shape[0]
        if n < WINDOW_SIZE:
            continue

        for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
            Xw = X[start:start + WINDOW_SIZE]
            X_list.append(window_features(Xw))
            y_list.append(activity)
            group_list.append(subject)

    return np.vstack(X_list), np.array(y_list), np.array(group_list)


# =========================
# MAIN
# =========================
def main():
    df = pd.read_parquet(PARQUET_PATH)

    # keep only selected activities
    df = df[df["activity"].isin(KEEP_ACTIVITIES)].copy()
    print("Rows after filter:", len(df))
    print(df["activity"].value_counts(), "\n")

    # keep only subjects that have all 3 activities (to make SGKF stable)
    subj_acts = df.groupby("subject_id")["activity"].unique()
    good_subjects = subj_acts[subj_acts.apply(lambda a: set(a) == set(KEEP_ACTIVITIES))].index.tolist()

    print("Subjects with ALL 3 activities:", len(good_subjects))
    if len(good_subjects) < N_SPLITS:
        raise RuntimeError(
            f"Need at least {N_SPLITS} subjects with all 3 activities for {N_SPLITS}-fold SGKF. "
            f"Found only {len(good_subjects)}."
        )

    df = df[df["subject_id"].isin(good_subjects)].copy()
    print("Rows after subject filter:", len(df))
    print("Remaining subjects:", df["subject_id"].nunique())
    print(df["activity"].value_counts(), "\n")

    sensor_cols = pick_sensor_columns(df)
    print("Sensor columns:", len(sensor_cols))

    le = LabelEncoder()
    le.fit(KEEP_ACTIVITIES)

    X, y_str, groups = build_windows(df, sensor_cols)
    y = le.transform(y_str)

    print("Windows:", X.shape)
    print("Window class counts:\n", pd.Series(y_str).value_counts(), "\n")
    print("Unique subjects:", len(np.unique(groups)), "\n")

    # SGKF split: subject-independent + class-balanced
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    folds = list(sgkf.split(X, y, groups=groups))
    train_idx, test_idx = folds[USE_FOLD_INDEX]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train, g_test = groups[train_idx], groups[test_idx]

    print("Train subjects:", len(np.unique(g_train)))
    print("Test subjects :", len(np.unique(g_test)))
    print("Subject overlap:", len(set(g_train).intersection(set(g_test))), "\n")

    print("Train class counts:\n", pd.Series(le.inverse_transform(y_train)).value_counts(), "\n")
    print("Test class counts:\n", pd.Series(le.inverse_transform(y_test)).value_counts(), "\n")

    clf = ExtraTreesClassifier(
        n_estimators=800,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt"
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)

    labels_all = np.arange(len(le.classes_))
    print("\nClassification report:\n",
          classification_report(y_test, preds, labels=labels_all, target_names=le.classes_, zero_division=0))

    print("\nConfusion matrix:\n", confusion_matrix(y_test, preds, labels=labels_all))


if __name__ == "__main__":
    main()
