import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from make_windows import make_windows 


# ==========================================
# MODEL
# ==========================================
class CNN_LSTM_Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ==========================================
# LABEL CREATION
# ==========================================
def extract_label(filename):
    filename = str(filename).lower()

    if "normal" in filename:
        return "normal"
    elif "parkinson" in filename:
        return "parkinsonian"
    elif "ataxic" in filename:
        return "ataxic"
    elif "antalgic" in filename:
        return "antalgic"
    else:
        return "unknown"


# ==========================================
# MAIN
# ==========================================
def main():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(ROOT, "..", "saved_models")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load dataset
    file_path = r"C:\Users\ashok\OneDrive\Desktop\gait\gait_dataset.parquet"
    df = pd.read_parquet(file_path)

    print("\n📊 Columns in dataset:")
    print(df.columns.tolist())

    # ------------------------------------------
    # CREATE LABEL COLUMN
    # ------------------------------------------
    df["label"] = df["source_file"].apply(extract_label)

    print("\n✅ Label distribution:")
    print(df["label"].value_counts())

    # ------------------------------------------
    # FEATURES
    # ------------------------------------------
    exclude_cols = ["subject_id", "source_file", "timestamp", "label"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"\nDetected {len(feature_cols)} sensor features.")

    # ------------------------------------------
    # WINDOWING
    # ------------------------------------------
    X_seq, y_seq = make_windows(
        df,
        feature_cols=feature_cols,
        label_col="label",
        group_col="subject_id",
        window=128,
        stride=64
    )

    print("\nWindowed shape:", X_seq.shape)

    # ------------------------------------------
    # ENCODING
    # ------------------------------------------
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_seq)

    print("\nClasses:", le.classes_)

    # ------------------------------------------
    # SCALING
    # ------------------------------------------
    N, T, F = X_seq.shape
    X_flat = X_seq.reshape(-1, F)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat).reshape(N, T, F)

    # ------------------------------------------
    # SPLIT
    # ------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    # ------------------------------------------
    # MODEL
    # ------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN_LSTM_Classifier(
        input_dim=len(feature_cols),
        num_classes=len(le.classes_)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------
    # TRAINING
    # ------------------------------------------
    epochs = 30
    batch_size = 32

    print("\n🚀 Training started...\n")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))

        total_loss = 0

        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)

            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(X_test.to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        acc = accuracy_score(y_test, preds)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Acc: {acc:.4f}")

    # ------------------------------------------
    # RESULTS
    # ------------------------------------------
    print("\nFinal Accuracy:", accuracy_score(y_test, preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds, target_names=le.classes_))

    # ------------------------------------------
    # SAVE
    # ------------------------------------------
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "lstm_sensor_model.pt"))
    joblib.dump(scaler, os.path.join(SAVE_DIR, "sensor_scaler.pkl"))
    joblib.dump(le, os.path.join(SAVE_DIR, "sensor_label_encoder.pkl"))

    print("\n✅ Model saved successfully!")


if __name__ == "__main__":
    main()