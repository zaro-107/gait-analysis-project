import os
import re
import numpy as np
import joblib
import torch
import torch.nn as nn

SPLIT_RE = re.compile(r"[,\t ]+")

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


def _load_numeric_matrix_from_bytes(file_bytes: bytes) -> np.ndarray:
    text = file_bytes.decode("utf-8", errors="ignore").splitlines()
    rows = []
    for line in text:
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
        raise ValueError("No numeric rows found in uploaded file.")

    max_cols = max(len(r) for r in rows)
    padded = []
    for r in rows:
        if len(r) < max_cols:
            r = r + [0.0] * (max_cols - len(r))
        padded.append(r)

    return np.array(padded, dtype=np.float32)


def _sliding_windows(X: np.ndarray, window_size: int, step: int):
    n = X.shape[0]
    out = []
    for start in range(0, n - window_size + 1, step):
        out.append(X[start:start + window_size])
    if not out:
        raise ValueError(f"Not enough rows for a window of size {window_size}.")
    return np.stack(out, axis=0)  # (Nw, T, C)


class ActivityPredictor:
    def __init__(self, save_dir: str = "saved_models", step: int = 128, embed_dim: int = 128):
        self.save_dir = save_dir
        self.step = step
        self.embed_dim = embed_dim

        meta_path = os.path.join(save_dir, "meta.joblib")
        clf_path = os.path.join(save_dir, "extratrees_on_embeddings.joblib")
        cnn_path = os.path.join(save_dir, "cnn_embedder.pt")

        self.meta = joblib.load(meta_path)
        self.clf = joblib.load(clf_path)

        self.mean = self.meta["mean"]
        self.std = self.meta["std"]
        self.classes = self.meta["classes"]
        self.window_size = int(self.meta["window_size"])
        self.C = len(self.mean)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = CNNEmbedder(in_ch=self.C, embed_dim=self.embed_dim).to(self.device)

        state = torch.load(cnn_path, map_location=self.device)

        # trained state came from CNNClassifier, keys likely "embedder.*"
        new_state = {}
        for k, v in state.items():
            if k.startswith("embedder."):
                new_state[k.replace("embedder.", "")] = v

        if new_state:
            self.embedder.load_state_dict(new_state, strict=True)
        else:
            self.embedder.load_state_dict(state, strict=False)

        self.embedder.eval()

    @torch.no_grad()
    def predict_bytes(self, file_bytes: bytes):
        X = _load_numeric_matrix_from_bytes(file_bytes)  # (T, C_file)

        # align columns to expected C
        if X.shape[1] >= self.C:
            X = X[:, :self.C]
        else:
            pad = np.zeros((X.shape[0], self.C - X.shape[1]), dtype=np.float32)
            X = np.concatenate([X, pad], axis=1)

        # normalize
        X = (X - self.mean) / self.std

        # windows
        W = _sliding_windows(X, window_size=self.window_size, step=self.step)  # (Nw, T, C)
        W_t = torch.from_numpy(W).float().transpose(1, 2).to(self.device)      # (Nw, C, T)

        Z = self.embedder(W_t).cpu().numpy()                                   # (Nw, embed_dim)
        pred_w = self.clf.predict(Z)

        counts = np.bincount(pred_w, minlength=len(self.classes))
        final = int(np.argmax(counts))

        vote_map = {str(self.classes[i]): int(counts[i]) for i in range(len(self.classes))}
        return str(self.classes[final]), vote_map
