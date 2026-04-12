# backend/predict_video_lstm.py

import os
import joblib
import numpy as np
import torch
import torch.nn as nn

from backend.feature_extractor import extract_pose_sequence


class VideoLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=6, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


MODEL_CACHE = {
    "model": None,
    "scaler": None,
    "label_encoder": None,
    "meta": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def load_video_lstm_once():
    if MODEL_CACHE["model"] is not None:
        return

    save_dir = "saved_models"
    model_path = os.path.join(save_dir, "video_lstm_gait.pt")
    scaler_path = os.path.join(save_dir, "video_lstm_scaler.pkl")
    le_path = os.path.join(save_dir, "video_lstm_label_encoder.pkl")
    meta_path = os.path.join(save_dir, "video_lstm_meta.pkl")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path) and os.path.exists(meta_path)):
        return

    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    meta = joblib.load(meta_path)

    model = VideoLSTMClassifier(
        input_dim=meta["feature_dim"],
        hidden_dim=meta["hidden_dim"],
        num_layers=meta["num_layers"],
        num_classes=len(meta["classes"]),
        dropout=meta["dropout"],
    ).to(MODEL_CACHE["device"])

    state = torch.load(model_path, map_location=MODEL_CACHE["device"])
    model.load_state_dict(state)
    model.eval()

    MODEL_CACHE["model"] = model
    MODEL_CACHE["scaler"] = scaler
    MODEL_CACHE["label_encoder"] = le
    MODEL_CACHE["meta"] = meta


def explain_gait_type(label, feats):
    cadence_l = float(feats.get("cadence_left_proxy_spm", 0.0))
    cadence_r = float(feats.get("cadence_right_proxy_spm", 0.0))
    cadence = (cadence_l + cadence_r) / 2.0

    step_var = float(feats.get("step_variability", 0.0))
    pelvis_sway = float(feats.get("pelvis_sway", 0.0))
    trunk_mean = float(feats.get("trunk_lean", {}).get("mean", 0.0))
    step_len = float(feats.get("step_length_proxy", {}).get("mean", 0.0))
    knee_sym = float(feats.get("symmetry", {}).get("knee_rom_0to1", 0.0))

    if label == "antalgic":
        return f"High asymmetry with reduced regularity suggests pain-avoiding walking. Knee symmetry={knee_sym:.3f}"
    if label == "spastic":
        return f"Reduced smoothness and stiff joint motion pattern suggest spastic gait."
    if label == "ataxic":
        return f"High step variability suggests unsteady gait. Step variability={step_var:.3f}"
    if label == "parkinsonian":
        return f"Short-step pattern and trunk posture suggest Parkinsonian gait. Step length proxy={step_len:.3f}, trunk lean={trunk_mean:.2f}"
    if label == "waddling":
        return f"Increased pelvic sway suggests waddling gait. Pelvis sway={pelvis_sway:.3f}"
    return f"Cadence average={cadence:.2f} spm, symmetry and sequence pattern closer to normal gait."


def predict_video_gait(video_path):
    load_video_lstm_once()

    if MODEL_CACHE["model"] is None:
        return {
            "status": "error",
            "message": "Video LSTM model not found. Train it first.",
        }

    target_frames = MODEL_CACHE["meta"]["target_frames"]

    seq, seq_meta = extract_pose_sequence(
        video_path=video_path,
        max_frames=300,
        sample_every=2,
        target_frames=target_frames,
    )

    scaler = MODEL_CACHE["scaler"]
    seq_scaled = scaler.transform(seq).astype(np.float32)
    x = torch.tensor(seq_scaled[None, :, :], dtype=torch.float32).to(MODEL_CACHE["device"])

    with torch.no_grad():
        logits = MODEL_CACHE["model"](x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    label = MODEL_CACHE["label_encoder"].inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])

    # Also compute summary features for explanation
    from backend.feature_extractor import extract_gait_features
    feats = extract_gait_features(video_path, return_series=False)

    return {
        "status": "ok",
        "model": "video_lstm_gait",
        "gait_type": label,
        "confidence": confidence,
        "classes": MODEL_CACHE["meta"]["classes"],
        "probabilities": {
            cls: float(p) for cls, p in zip(MODEL_CACHE["meta"]["classes"], probs)
        },
        "sequence_meta": seq_meta,
        "reason": explain_gait_type(label, feats),
        "summary_features": feats,
    }