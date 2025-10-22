from pydantic import BaseModel
from typing import List, Optional, Literal
from datetime import datetime

class ConfigModel(BaseModel):
    fps: int = 8
    enable_violence: bool = True
    enable_fire: bool = True
    violence_sequence_length: int = 16
    violence_threshold: float = 0.80
    violence_ema_alpha: float = 0.7  # smoothing factor for EMA of score
    violence_sustain_frames: int = 6  # consecutive frames >= threshold to trigger
    violence_release_frames: int = 3  # consecutive frames < threshold to clear
    violence_motion_min_ratio: float = 0.04  # require basic motion for violence alert
    enable_person_gate: bool = True
    person_require_proximity: bool = False
    person_conf_threshold: float = 0.35
    person_every_n: int = 3
    person_model_name: str = "yolov5s"
    person_infer_size: int = 512
    violence_min_persons: int = 2
    person_ttl_seconds: float = 1.0  # cache person detections for this time window
    person_max_pair_distance_ratio: float = 0.15  # fraction of frame diagonal for proximity
    person_min_pair_iou: float = 0.12  # alternatively, minimum IoU between two persons
    fire_threshold: float = 0.62
    fire_min_area: int = 1000  # min bbox area in pixels to count
    fire_sustain_frames: int = 4  # consecutive frames with fire to trigger
    fire_release_frames: int = 2  # consecutive empty frames to clear
    fire_allowed_labels: list[str] = ["fire"]
    fire_color_gate_enabled: bool = True
    fire_color_min_ratio: float = 0.20
    fire_color_min_ratio_small: float = 0.14
    fire_color_small_area: int = 900
    fire_iou_min: float = 0.40
    fire_infer_size: int = 640
    fire_every_n: int = 3
    fire_backend: str = "yolov5"
    fire_motion_min_ratio: float = 0.0
    # Secondary violence model (optional re-ranker/ensemble)
    enable_violence_secondary: bool = False
    violence_secondary_backend: str = "onnx"
    violence_secondary_model_path: str = "models/violence_secondary.onnx"
    violence_secondary_T: int = 16
    violence_secondary_input_size: int = 224
    violence_fusion_mode: str = "confirm"  # confirm | average
    violence_fusion_weight_secondary: float = 0.5
    violence_secondary_threshold: float = 0.7
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