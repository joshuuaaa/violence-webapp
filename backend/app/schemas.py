from pydantic import BaseModel
from typing import List, Optional, Literal
from datetime import datetime

class ConfigModel(BaseModel):
    fps: int = 5
    enable_violence: bool = True
    enable_fire: bool = True
    violence_sequence_length: int = 16
    violence_threshold: float = 0.6
    fire_threshold: float = 0.4
    violence_model_path: str = "models/violence_model.keras"
    fire_model_weights: str = "models/best.pt"
    recording_enabled: bool = True
    pre_roll_seconds: int = 3
    post_roll_seconds: int = 5
    recordings_dir: str = "recordings"

class DetectionBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float

class InferenceResult(BaseModel):
    violence_score: Optional[float] = None
    fire_boxes: Optional[List[DetectionBox]] = None
    alert: bool = False
    alert_types: List[Literal["violence","fire"]] = []

class EventResponse(BaseModel):
    id: int
    event_type: str
    confidence: float
    timestamp: datetime
    snapshot_path: Optional[str] = None
    video_path: Optional[str] = None

class StatsResponse(BaseModel):
    total: int
    by_type: dict
    by_day: dict