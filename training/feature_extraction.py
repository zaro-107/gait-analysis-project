import numpy as np
import pandas as pd
import scipy.fftpack

def extract_features(window):
    """
    Extracts time-domain and frequency-domain features for each sensor column.
    """
    feats = []

    for col in window.columns:
        data = window[col].values

        # 1. Time-Domain Features (Your original 9 features)
        time_feats = [
            np.mean(data),
            np.std(data),
            np.min(data),
            np.max(data),
            np.median(data),
            np.var(data),
            np.sum(data ** 2),           # energy
            np.percentile(data, 25),
            np.percentile(data, 75),
        ]
        
        # 2. Frequency-Domain Features (New: Spectral Analysis for Gait Rhythm)
        # Handle cases where data might be completely flat (e.g., sensor error)
        if np.var(data) > 0:
            fft_vals = np.abs(scipy.fftpack.fft(data))
            # Skip the DC component (index 0) to find the dominant movement frequency
            dominant_freq = np.argmax(fft_vals[1:]) + 1 
        else:
            dominant_freq = 0.0

        feats.extend(time_feats + [dominant_freq])

    return feats


def create_dataset(df, window_size=128, step=64):
    """
    Slides a window over the raw sensor data and builds a feature matrix.
    """
    X = []
    y = []
    groups = []

    # FIX: Dynamically identify sensor columns instead of hardcoding '0' through '38'
    exclude_cols = ["subject_id", "source_file", "label", "timestamp"]
    sensor_cols = [col for col in df.columns if col not in exclude_cols]
    
    if len(sensor_cols) == 0:
        raise ValueError("No sensor columns found! Check your raw CSV headers.")

    for file_name, group in df.groupby("source_file"):
        group = group.reset_index(drop=True)

        # Safer label extraction
        parts = str(file_name).split("_")
        if len(parts) > 2:
            label = parts[2].lower()
        else:
            label = "unknown"

        # Slide window
        for i in range(0, len(group) - window_size + 1, step):
            window = group.iloc[i:i + window_size][sensor_cols]
            
            # Extract engineered features
            features = extract_features(window)
            
            X.append(features)
            y.append(label)
            groups.append(file_name)

    return np.array(X), np.array(y), np.array(groups)