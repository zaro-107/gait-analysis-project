import os
import re
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

SPLIT_RE = re.compile(r"[,\t ]+")

SAVE_DIR = "saved_models"
TXT_PATH = r"C:\Users\ashok\Downloads\HumanGaitDataBase\Data\HuGaDB_v1_walking_01_00.txt"

def load_numeric_matrix(path: str) -> np.ndarray:
    rows = []
    try:
        lines = open(path, "r", encoding="utf-8", errors="ignore").read().splitlines()
    except Exception:
        lines = open(path, "r", encoding="latin-1", errors="ignore").read().splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = [p for p in SPLIT_RE.split(line) if p != ""]
        if len(parts) < 3:
            continue
        try:
            row = [float(x) for x in parts]
        except ValueError:
            continue
        rows.append(row)

    if not rows:
        raise RuntimeError("No numeric rows found in file")

    # pad ragged rows
    max_cols = max(len(r) for r in rows)
    padded = []
    for r in rows:
        if len(r) < max_cols:
            r = r + [0.0] * (max_cols - len(r))
        padded.append(r)

    return np.array(padded, dtype=np.float32)

class CNNEmbedder(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.proj(x)
        return x

def sliding_windows(X, window_size=256, step=128):
    out = []
    n = X.shape[0]
    for start in range(0, n - window_size + 1, step):
        out.append(X[start:start+window_size])
    if not out:
        raise RuntimeError("Not enough rows for a window.")
    return np.stack(out, axis=0)  # (Nw, T, C)

def main():
    meta = joblib.load(os.path.join(SAVE_DIR, "meta.joblib"))
    clf = joblib.load(os.path.join(SAVE_DIR, "extratrees_on_embeddings.joblib"))

    mean = meta["mean"]
    std = meta["std"]
    classes = meta["classes"]
    window_size = meta["window_size"]

    X = load_numeric_matrix(TXT_PATH)  # (T, C_file)
    # Use first 39 columns if file has more; pad if fewer
    C = len(mean)
    if X.shape[1] >= C:
        X = X[:, :C]
    else:
        pad = np.zeros((X.shape[0], C - X.shape[1]), dtype=np.float32)
        X = np.concatenate([X, pad], axis=1)

    X = (X - mean) / std

    W = sliding_windows(X, window_size=window_size, step=128)  # (Nw, T, C)
    W_t = torch.from_numpy(W).float().transpose(1, 2)          # (Nw, C, T)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = CNNEmbedder(in_ch=C, embed_dim=128).to(device)
    state = torch.load(os.path.join(SAVE_DIR, "cnn_embedder.pt"), map_location=device)
    # state dict belongs to CNNClassifier; extract embedder weights by key prefix
    # If keys include "embedder.", strip it; else load directly if matching
    new_state = {}
    for k, v in state.items():
        if k.startswith("embedder."):
            new_state[k.replace("embedder.", "")] = v
    if new_state:
        embedder.load_state_dict(new_state, strict=True)
    else:
        embedder.load_state_dict(state, strict=False)

    embedder.eval()
    with torch.no_grad():
        Z = embedder(W_t.to(device)).cpu().numpy()  # (Nw, 128)

    # window-level predictions -> majority vote
    pred_w = clf.predict(Z)
    counts = np.bincount(pred_w, minlength=len(classes))
    final = np.argmax(counts)

    print("Predicted activity:", classes[final])
    print("Vote counts:", {classes[i]: int(counts[i]) for i in range(len(classes))})

if __name__ == "__main__":
    main()
