import os
import sys
import glob
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --------------------------------------------------
# 1. PATH SETUP & IMPORTS
# --------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../gait/training
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                # .../gait

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.feature_extractor import extract_pose_sequence

DATA_DIR = os.path.join(PROJECT_ROOT, "video_data")
SAVE_DIR = os.path.join(PROJECT_ROOT, "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVE_DIR, "video_lstm_gait.pt")
SCALER_PATH = os.path.join(SAVE_DIR, "video_lstm_scaler.pkl")
LE_PATH = os.path.join(SAVE_DIR, "video_lstm_label_encoder.pkl")
META_PATH = os.path.join(SAVE_DIR, "video_lstm_meta.pkl")

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------
TARGET_FRAMES = 60
BATCH_SIZE = 16
EPOCHS = 60
LR = 0.0005

# Dynamically find ALL folders inside the video_data directory
CLASSES_TO_TRAIN = [
    d for d in os.listdir(DATA_DIR) 
    if os.path.isdir(os.path.join(DATA_DIR, d))
]
# --------------------------------------------------
# 3. PYTORCH DATASET & MODEL
# --------------------------------------------------
class GaitSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
        )

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

# --------------------------------------------------
# 4. DATA EXTRACTION FROM FOLDERS
# --------------------------------------------------
def build_sequence_dataset():
    X_list, y_list = [], []
    
    print(f"=== Extracting sequences from: {DATA_DIR} ===")
    
    for class_name in CLASSES_TO_TRAIN:
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        videos = glob.glob(os.path.join(class_dir, "*.mp4")) + glob.glob(os.path.join(class_dir, "*.avi"))
        
        print(f"Processing '{class_name}': found {len(videos)} videos")
        
        for vid_path in videos:
            try:
                # Extract 60 frames of 3D poses (Shape: [60, Feature_Dim])
                seq, meta = extract_pose_sequence(
                    video_path=vid_path,
                    max_frames=300,
                    sample_every=2,
                    target_frames=TARGET_FRAMES,
                    timeout_sec=30
                )
                
                # Check for bad extraction
                if seq is not None and not np.any(np.isnan(seq)):
                    X_list.append(seq)
                    # Convert "normal_gait" -> "normal" to match backend expectations
                    clean_label = class_name.replace("_gait", "")
                    y_list.append(clean_label)
            except Exception as e:
                print(f"  Skipped {os.path.basename(vid_path)}: {e}")

    if not X_list:
        raise RuntimeError("No usable video sequences extracted. Check video_data folder.")

    return np.array(X_list), np.array(y_list)

# --------------------------------------------------
# 5. TRAINING PIPELINE
# --------------------------------------------------
def train():
    # 1. Extract Data
    X_raw, y_raw = build_sequence_dataset()
    print(f"\nExtracted dataset shape: X={X_raw.shape}, y={y_raw.shape}")
    
    num_samples, seq_len, feature_dim = X_raw.shape

    # 2. Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    print(f"Classes found: {list(le.classes_)}")

    # 3. Scale Features (Flatten, scale, then reshape back to sequences)
    scaler = StandardScaler()
    X_flat = X_raw.reshape(-1, feature_dim)
    X_flat_scaled = scaler.fit_transform(X_flat)
    X_scaled = X_flat_scaled.reshape(num_samples, seq_len, feature_dim)

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    train_loader = DataLoader(GaitSequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(GaitSequenceDataset(X_test, y_test), batch_size=BATCH_SIZE)

    # 5. Initialize Model
    model = VideoLSTMClassifier(input_dim=feature_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("\n🚀 Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()

        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%")

    # 6. Save Artifacts
    print("\n💾 Saving model artifacts...")
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, LE_PATH)

    meta = {
        "feature_dim": feature_dim,
        "target_frames": TARGET_FRAMES,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "classes": list(le.classes_),
    }
    joblib.dump(meta, META_PATH)
    print("✅ Training complete! Models saved to /saved_models.")

if __name__ == "__main__":
    train()