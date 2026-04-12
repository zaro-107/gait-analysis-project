import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader


# =========================
# CONFIG
# =========================
PARQUET_PATH = "hugodab_dataset.parquet"
KEEP_ACTIVITIES = ["walking", "sitting", "standing"]  # stable subject-independent set

WINDOW_SIZE = 256
STEP_SIZE = 128

N_SPLITS = 5
USE_FOLD_INDEX = 0

BATCH_SIZE = 128
EPOCHS = 8
LR = 1e-3
EMBED_DIM = 128

SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# DATA HELPERS
# =========================
def pick_sensor_columns(df: pd.DataFrame):
    meta = {"activity", "subject_id", "trial_id", "source_file"}
    return [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]

def build_windows_raw(df: pd.DataFrame, sensor_cols):
    """
    Returns:
      X_raw: (N, WINDOW_SIZE, C)
      y_str: (N,)
      groups: (N,) subject_id per window
    """
    X_list, y_list, group_list = [], [], []

    for _, g in df.groupby("source_file", sort=False):
        activity = g["activity"].iloc[0]
        subject = g["subject_id"].iloc[0]

        X = g[sensor_cols].to_numpy(dtype=np.float32)
        n = X.shape[0]
        if n < WINDOW_SIZE:
            continue

        for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
            Xw = X[start:start + WINDOW_SIZE]
            # NaN -> 0 (from padding)
            Xw = np.nan_to_num(Xw, nan=0.0)
            X_list.append(Xw)
            y_list.append(activity)
            group_list.append(subject)

    X_raw = np.stack(X_list, axis=0)  # (N, T, C)
    y_str = np.array(y_list)
    groups = np.array(group_list)
    return X_raw, y_str, groups


class WindowDataset(Dataset):
    def __init__(self, X, y):
        """
        X: (N, T, C) float32
        y: (N,) int64
        """
        self.X = X
        self.y = y

        # normalize per-channel using train stats later; here keep raw
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]                 # (T, C)
        x = torch.from_numpy(x).float() # float32
        y = torch.tensor(self.y[idx]).long()
        # CNN expects (C, T)
        return x.transpose(0, 1), y


# =========================
# MODEL
# =========================
class CNNEmbedder(nn.Module):
    """
    1D CNN that outputs an embedding vector (EMBED_DIM).
    Input: (B, C, T)
    """
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

            nn.AdaptiveAvgPool1d(1),  # (B, 256, 1)
        )
        self.proj = nn.Sequential(
            nn.Flatten(),             # (B, 256)
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.proj(x)
        return x


class CNNClassifier(nn.Module):
    """
    For training embedder: embedding -> linear classifier
    """
    def __init__(self, in_ch: int, num_classes: int, embed_dim: int = 128):
        super().__init__()
        self.embedder = CNNEmbedder(in_ch, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        z = self.embedder(x)
        logits = self.head(z)
        return logits, z


# =========================
# TRAIN + EMBEDDING EXTRACTION
# =========================
def compute_channel_norm_stats(X_train_raw):
    """
    X_train_raw: (N, T, C)
    returns mean(C,), std(C,)
    """
    # flatten over N and T
    flat = X_train_raw.reshape(-1, X_train_raw.shape[-1])  # (N*T, C)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)

def apply_channel_norm(X_raw, mean, std):
    return (X_raw - mean) / std

@torch.no_grad()
def extract_embeddings(model: CNNClassifier, loader: DataLoader, device: str):
    model.eval()
    Z = []
    Y = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits, z = model(xb)
        Z.append(z.cpu().numpy())
        Y.append(yb.numpy())
    return np.vstack(Z), np.concatenate(Y)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["activity"].isin(KEEP_ACTIVITIES)].copy()

    # keep only subjects with all 3 activities (stable SGKF)
    subj_acts = df.groupby("subject_id")["activity"].unique()
    good_subjects = subj_acts[subj_acts.apply(lambda a: set(a) == set(KEEP_ACTIVITIES))].index.tolist()
    df = df[df["subject_id"].isin(good_subjects)].copy()

    print("Subjects used:", df["subject_id"].nunique())
    print(df["activity"].value_counts(), "\n")

    sensor_cols = pick_sensor_columns(df)
    C = len(sensor_cols)
    print("Sensor channels:", C)

    # window raw
    X_raw, y_str, groups = build_windows_raw(df, sensor_cols)
    print("Windows raw:", X_raw.shape, "Unique subjects:", len(np.unique(groups)))

    le = LabelEncoder()
    le.fit(KEEP_ACTIVITIES)
    y = le.transform(y_str)

    # SGKF split
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    folds = list(sgkf.split(X_raw, y, groups=groups))
    train_idx, test_idx = folds[USE_FOLD_INDEX]

    X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train, g_test = groups[train_idx], groups[test_idx]

    print("\nTrain subjects:", len(np.unique(g_train)), "Test subjects:", len(np.unique(g_test)))
    print("Subject overlap:", len(set(g_train).intersection(set(g_test))))

    # normalize per channel using TRAIN ONLY
    mean, std = compute_channel_norm_stats(X_train_raw)
    X_train = apply_channel_norm(X_train_raw, mean, std)
    X_test = apply_channel_norm(X_test_raw, mean, std)

    # dataloaders
    train_ds = WindowDataset(X_train, y_train)
    test_ds = WindowDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # CNN
    model = CNNClassifier(in_ch=C, num_classes=len(le.classes_), embed_dim=EMBED_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # train
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item() * yb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # quick eval
        model.eval()
        with torch.no_grad():
            correct_t = 0
            total_t = 0
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits, _ = model(xb)
                preds = torch.argmax(logits, dim=1)
                correct_t += (preds == yb).sum().item()
                total_t += yb.size(0)
            test_acc = correct_t / total_t

        model.train()
        print(f"Epoch {epoch}/{EPOCHS} | loss {train_loss:.4f} | train_acc {train_acc:.4f} | test_acc {test_acc:.4f}")

    # extract embeddings
    Z_train, y_train_out = extract_embeddings(model, train_loader, device)
    Z_test, y_test_out = extract_embeddings(model, test_loader, device)

    # train classical classifier on embeddings (HYBRID)
    clf = ExtraTreesClassifier(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt",
    )
    clf.fit(Z_train, y_train_out)

    preds = clf.predict(Z_test)
    acc = accuracy_score(y_test_out, preds)
    print("\nHYBRID (CNN embeddings -> ExtraTrees) Accuracy:", acc)
    print("\nReport:\n", classification_report(y_test_out, preds, target_names=le.classes_, zero_division=0))
    print("\nConfusion matrix:\n", confusion_matrix(y_test_out, preds))

    # save everything for inference
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "cnn_embedder.pt"))
    joblib.dump(clf, os.path.join(SAVE_DIR, "extratrees_on_embeddings.joblib"))
    joblib.dump({"mean": mean, "std": std, "classes": le.classes_, "sensor_cols": sensor_cols,
                 "window_size": WINDOW_SIZE}, os.path.join(SAVE_DIR, "meta.joblib"))

    print("\n Saved to:", SAVE_DIR)
    print(" - cnn_embedder.pt")
    print(" - extratrees_on_embeddings.joblib")
    print(" - meta.joblib")


if __name__ == "__main__":
    main()
