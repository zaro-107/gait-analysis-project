import io
import os
import sys
import uuid
import traceback
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from typing import Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ==========================================
# PATH FIX: Ensure the main 'gait' folder is in path
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# =========================
# FIXED LOCAL IMPORTS
# =========================
from backend.utils.risk_engine import calculate_risk, get_severity, doctor_recommendation
from backend.utils.report_generator import generate_report

from backend.feature_extractor import extract_gait_features
from backend.predict_video_lstm import predict_video_gait

# =========================
# DATABASE (OPTIONAL)
# =========================
try:
    from sqlalchemy.orm import Session
    from backend.database import engine, Base, get_db
    from backend.models_db import PredictionRecord
    
    Base.metadata.create_all(bind=engine)
    DB_ENABLED = True
except:
    DB_ENABLED = False
    def get_db():
        yield None
    Session = Any

# =========================
# APP INIT
# =========================
app = FastAPI(title="Gait Analysis API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# ROUTES
# =========================

@app.get("/download_report/{file_name}")
async def download_report(file_name: str):
    path = os.path.join(UPLOAD_DIR, file_name)
    if os.path.exists(path):
        return FileResponse(path, filename="Gait_Report.pdf")
    raise HTTPException(status_code=404, detail="Report not found")


@app.post("/predict_media")
async def predict_media(file: UploadFile = File(...), db: Session = Depends(get_db)):

    ext = os.path.splitext(file.filename)[1].lower()
    allowed = [".mp4", ".avi", ".mov", ".mkv"]

    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Save file
    filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, filename)

    with open(path, "wb") as f:
        f.write(await file.read())

    try:
        # =========================
        # 1. FEATURE EXTRACTION
        # =========================
        rich_feats = extract_gait_features(path, max_frames=120, return_series=True)

        # =========================
        # 2. MODEL PREDICTION
        # =========================
        lstm = predict_video_gait(path)

        # =========================
        # 3. DERIVED METRICS
        # =========================
        conf = lstm.get("confidence", 0)

        step_len = rich_feats.get("step_length_proxy", {}).get("mean", 0)

        cad_l = rich_feats.get("cadence_left_proxy_spm", 0)
        cad_r = rich_feats.get("cadence_right_proxy_spm", 0)
        avg_cadence = (cad_l + cad_r) / 2

        symmetry = rich_feats.get("symmetry", {}).get("hip_rom_0to1", 0)

        # =========================
        # 4. RISK ENGINE
        # =========================
        risk = calculate_risk(conf, step_len, avg_cadence, symmetry)
        severity = get_severity(risk)
        recs = doctor_recommendation(severity)

        # =========================
        # 5. REPORT GENERATION
        # =========================
        report_data = {
            "filename": file.filename,
            "medical_analysis": {
                "risk_score": risk,
                "severity": severity,
                "recommendation": recs
            }
        }

        report_name = f"report_{uuid.uuid4().hex}.pdf"
        report_path = os.path.join(UPLOAD_DIR, report_name)

        generate_report(report_data, report_path)

        # =========================
        # 6. DATABASE SAVE (OPTIONAL)
        # =========================
        db_id = None
        if DB_ENABLED and db:
            record = PredictionRecord(
                patient_id="anonymous",
                gait_type=lstm.get("gait_type"),
                confidence=conf,
                features=rich_feats
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            db_id = record.id

        # =========================
        # 7. FINAL RESPONSE (FIXED UI ISSUE)
        # =========================
        return {
            "status": "ok",
            "type": "video",
            "filename": file.filename,
            "db_record_id": db_id,

            #  THESE FIX EMPTY UI
            "frames_used": rich_feats.get("meta", {}).get("frames_raw_used"),
            "fps_effective": rich_feats.get("meta", {}).get("fps_effective"),
            "avg_cadence": avg_cadence,
            "symmetry_index": symmetry,

            "predictions": {
                "video_lstm_gait_type": lstm
            },

            "gait_features_rich": rich_feats,
            "report_file": report_name
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(path):
            os.remove(path)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)