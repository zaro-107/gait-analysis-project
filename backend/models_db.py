from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from datetime import datetime
from backend.database import Base

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    gait_type = Column(String)
    confidence = Column(Float)
    features = Column(JSON) # Store the rich features here
    rule_hints = Column(JSON)