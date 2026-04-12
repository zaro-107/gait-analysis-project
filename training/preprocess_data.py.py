import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

WINDOW_SIZE = 128
STEP_SIZE = 64


def load_data(file_path):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".txt"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    print("Loaded shape:", df.shape)
    return df


def detect_label_column(df):
    possible_labels = ["label", "activity", "class", "target", "action"]
    for col in df.columns:
        if col.lower() in possible_labels:
            return col
    return None


def get_sensor_columns(df):
    sensor_cols = []

    for col in df.columns:
        c = col.lower()
        if any(key in c for key in ["acc", "gyro", "ankle", "thigh", "shank", "imu"]):
            if pd.api.types.is_numeric_dtype(df[col]):
                sensor_cols.append(col)

    if not sensor_cols:
        # fallback: use all numeric columns except obvious metadata/labels
        exclude_words = ["label", "activity", "class", "target", "subject", "id", "trial", "time"]
        for col in df.columns:
            c = col.lower()
            if pd.api.types.is_numeric_dtype(df[col]) and not any(w in c for w in exclude_words):
                sensor_cols.append(col)

    if not sensor_cols:
        raise ValueError("No usable sensor columns found.")

    print("\nDetected sensor columns:")
    print(sensor_cols)
    return sensor_cols


def clean_data(df, sensor_cols):
    df = df.copy()

    df[sensor_cols] = df[sensor_cols].replace([np.inf, -np.inf], np.nan)
    df[sensor_cols] = df[sensor_cols].ffill().bfill()

    df = df.dropna(subset=sensor_cols)

    print("After cleaning shape:", df.shape)
    return df


def normalize_data(df, sensor_cols):
    scaler = StandardScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    return df, scaler


def create_windows(df, sensor_cols, label_col=None):
    X = []
    y = []

    values = df[sensor_cols].values

    for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window = values[start:end]
        X.append(window)

        if label_col is not None:
            label = df[label_col].iloc[start:end].mode()[0]
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y) if label_col is not None else None

    print("Windowed X shape:", X.shape)
    if y is not None:
        print("Windowed y shape:", y.shape)

    return X, y


def preprocess(file_path, save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)

    df = load_data(file_path)

    label_col = detect_label_column(df)
    print("Detected label column:", label_col)

    sensor_cols = get_sensor_columns(df)
    df = clean_data(df, sensor_cols)
    df, scaler = normalize_data(df, sensor_cols)
    X, y = create_windows(df, sensor_cols, label_col)

    np.savez(os.path.join(save_dir, "processed_data.npz"), X=X, y=y)
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

    meta = {
        "file_path": file_path,
        "label_col": label_col,
        "sensor_cols": sensor_cols,
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "num_samples": int(X.shape[0]),
        "num_features": int(X.shape[2]) if X.ndim == 3 else 0,
    }

    joblib.dump(meta, os.path.join(save_dir, "metadata.pkl"))

    print("\nSaved files:")
    print(os.path.join(save_dir, "processed_data.npz"))
    print(os.path.join(save_dir, "scaler.pkl"))
    print(os.path.join(save_dir, "metadata.pkl"))

    return X, y, scaler, meta


if __name__ == "__main__":
    FILE_PATH = r"C:\Users\ashok\OneDrive\Desktop\gait\gait_dataset.parquet"
    preprocess(FILE_PATH)