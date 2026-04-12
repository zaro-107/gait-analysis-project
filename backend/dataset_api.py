import os
import uuid
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

router = APIRouter(prefix="/dataset", tags=["dataset"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "trained_models")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def _safe_read_csv(path: str) -> pd.DataFrame:
    # Handles common CSV issues
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV read failed: {e}")


def _dataset_summary(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    rows, cols = df.shape
    missing = df.isna().sum().to_dict()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    numeric_stats = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().T  # count, mean, std, min, 25%, 50%, 75%, max
        for col in numeric_cols:
            numeric_stats[col] = {
                "count": float(desc.loc[col, "count"]),
                "mean": float(desc.loc[col, "mean"]) if not np.isnan(desc.loc[col, "mean"]) else None,
                "std": float(desc.loc[col, "std"]) if not np.isnan(desc.loc[col, "std"]) else None,
                "min": float(desc.loc[col, "min"]) if not np.isnan(desc.loc[col, "min"]) else None,
                "p25": float(desc.loc[col, "25%"]) if not np.isnan(desc.loc[col, "25%"]) else None,
                "median": float(desc.loc[col, "50%"]) if not np.isnan(desc.loc[col, "50%"]) else None,
                "p75": float(desc.loc[col, "75%"]) if not np.isnan(desc.loc[col, "75%"]) else None,
                "max": float(desc.loc[col, "max"]) if not np.isnan(desc.loc[col, "max"]) else None,
            }

    target_info = None
    if target_col and target_col in df.columns:
        vc = df[target_col].value_counts(dropna=False)
        target_info = {
            "target_col": target_col,
            "class_counts": {str(k): int(v) for k, v in vc.items()},
            "num_classes": int(vc.shape[0]),
        }

    return {
        "shape": {"rows": int(rows), "cols": int(cols)},
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "non_numeric_columns": non_numeric_cols,
        "missing_values": {k: int(v) for k, v in missing.items()},
        "numeric_stats": numeric_stats,
        "target_info": target_info,
    }


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target_col: Optional[str] = None
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    ds_id = uuid.uuid4().hex
    safe_name = file.filename.replace(" ", "_")
    out_path = os.path.join(DATASET_DIR, f"{ds_id}_{safe_name}")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty CSV uploaded")

    with open(out_path, "wb") as f:
        f.write(content)

    df = _safe_read_csv(out_path)
    summary = _dataset_summary(df, target_col=target_col)

    return {
        "dataset_id": ds_id,
        "saved_as": os.path.basename(out_path),
        "summary": summary,
    }


@router.post("/train")
async def train_model(
    dataset_id: str,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    # Find dataset file by prefix
    matches = [fn for fn in os.listdir(DATASET_DIR) if fn.startswith(dataset_id + "_")]
    if not matches:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload first.")
    path = os.path.join(DATASET_DIR, matches[0])

    df = _safe_read_csv(path)

    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"target_col '{target_col}' not found in columns")

    # Keep only numeric features for baseline model
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No numeric feature columns found. Provide numeric features.")

    # Drop rows with missing in X/y
    data = pd.concat([X_num, y], axis=1).dropna()
    X_num = data.drop(columns=[target_col])
    y = data[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_num, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )

    # Simple strong baseline (you can swap to XGBoost later)
    clf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    model_path = os.path.join(MODEL_DIR, f"{dataset_id}_model.joblib")
    joblib.dump(
        {"model": clf, "feature_columns": X_num.columns.tolist(), "target_col": target_col},
        model_path
    )

    return {
        "dataset_id": dataset_id,
        "target_col": target_col,
        "num_features_used": int(X_num.shape[1]),
        "accuracy": acc,
        "classification_report": report,
        "saved_model": os.path.basename(model_path),
    }
