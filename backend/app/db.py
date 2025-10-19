from datetime import datetime
from typing import Optional, List

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///detections.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String, index=True)  # "violence" | "fire"
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    snapshot_path = Column(String, nullable=True)
    video_path = Column(String, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)