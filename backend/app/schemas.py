from pydantic import BaseModel
from typing import List, Optional, Literal
from datetime import datetime

class ConfigModel(BaseModel):
    fps: int = 5
    enable_violence: bool = True
    enable_fire: bool = True
    violence_sequence_length: int = 16
    violence_threshold: float = 0.8
    violence_ema_alpha: float = 0.6  # smoothing factor for EMA of score
    violence_sustain_frames: int = 5  # consecutive frames >= threshold to trigger
    violence_release_frames: int = 3  # consecutive frames < threshold to clear
    violence_motion_min_ratio: float = 0.02  # require basic motion for violence alert
    enable_person_gate: bool = True
    person_conf_threshold: float = 0.35
    person_every_n: int = 3
    person_model_name: str = "yolov5n"
    person_infer_size: int = 416
    violence_min_persons: int = 2
    person_ttl_seconds: float = 1.0  # cache person detections for this time window
    person_max_pair_distance_ratio: float = 0.18  # fraction of frame diagonal for proximity
    person_min_pair_iou: float = 0.05  # alternatively, minimum IoU between two persons
    fire_threshold: float = 0.4
    fire_min_area: int = 750  # min bbox area in pixels to count
    fire_sustain_frames: int = 2  # consecutive frames with fire to trigger
    fire_release_frames: int = 2  # consecutive empty frames to clear
    fire_allowed_labels: list[str] = ["fire"]
    fire_color_gate_enabled: bool = True
    fire_color_min_ratio: float = 0.25
    fire_color_min_ratio_small: float = 0.12
    fire_color_small_area: int = 800
    fire_iou_min: float = 0.3
    fire_infer_size: int = 512
    fire_every_n: int = 2
    fire_backend: str = "yolov5"
    fire_motion_min_ratio: float = 0.0
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