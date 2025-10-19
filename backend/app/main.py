import os
import io
import json
from datetime import datetime
from typing import Dict, Any, List

import yaml
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .db import init_db, SessionLocal, DetectionEvent
from .schemas import ConfigModel, EventResponse, StatsResponse
from .inference import InferenceEngine
from .services.recorder import IncidentRecorder

CONFIG_PATH = "config/config.yaml"
DEFAULT_CONFIG = {
    "fps": 5,
    "enable_violence": True,
    "enable_fire": True,
    "violence_sequence_length": 16,
    "violence_threshold": 0.6,
    "fire_threshold": 0.4,
    "violence_model_path": "models/violence_model.keras",
    "fire_model_weights": "models/best.pt",
    "recording_enabled": True,
    "pre_roll_seconds": 3,
    "post_roll_seconds": 5,
    "recordings_dir": "recordings",
}

def load_config() -> Dict[str, Any]:
    os.makedirs("config", exist_ok=True)
    if not os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            yaml.safe_dump(DEFAULT_CONFIG, f)
        return DEFAULT_CONFIG.copy()
    with open(CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f) or {}
    merged = {**DEFAULT_CONFIG, **data}
    return merged

def save_config(cfg: Dict[str, Any]):
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f)

app = FastAPI(title="Smart CCTV")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()
cfg = load_config()
init_db()
os.makedirs("snapshots", exist_ok=True)
os.makedirs(cfg.get("recordings_dir", "recordings"), exist_ok=True)

# Lazy init to fail fast if models missing
engine = InferenceEngine(cfg)
recorder = IncidentRecorder(
    output_dir=cfg.get("recordings_dir", "recordings"),
    fps=cfg.get("fps", 5),
    frame_size=(640, 480),
    pre_roll=cfg.get("pre_roll_seconds", 3),
    post_roll=cfg.get("post_roll_seconds", 5),
)

# Static mounts for frontend and generated media
app.mount("/static", StaticFiles(directory="frontend"), name="static")
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")
app.mount("/recordings", StaticFiles(directory="recordings"), name="recordings")

@app.get("/")
def root_index():
    index_path = os.path.join("frontend", "index.html")
    return FileResponse(index_path)

@app.get("/api/config")
def get_config():
    return cfg

@app.post("/api/config")
def update_config(new_cfg: ConfigModel):
    global cfg, engine, recorder
    cfg = new_cfg.model_dump()
    save_config(cfg)
    # Re-init models and recorder with new config
    engine = InferenceEngine(cfg)
    recorder = IncidentRecorder(
        output_dir=cfg.get("recordings_dir", "recordings"),
        fps=cfg.get("fps", 5),
        frame_size=(640, 480),
        pre_roll=cfg.get("pre_roll_seconds", 3),
        post_roll=cfg.get("post_roll_seconds", 5),
    )
    return {"ok": True}

@app.get("/api/events", response_model=List[EventResponse])
def list_events():
    db = SessionLocal()
    try:
        rows = db.query(DetectionEvent).order_by(DetectionEvent.timestamp.desc()).limit(1000).all()
        return [
            EventResponse(
                id=r.id,
                event_type=r.event_type,
                confidence=r.confidence,
                timestamp=r.timestamp,
                snapshot_path=r.snapshot_path,
                video_path=r.video_path
            ) for r in rows
        ]
    finally:
        db.close()

@app.get("/api/stats", response_model=StatsResponse)
def stats():
    db = SessionLocal()
    try:
        rows = db.query(DetectionEvent).all()
        by_type = {}
        by_day = {}
        for r in rows:
            by_type[r.event_type] = by_type.get(r.event_type, 0) + 1
            day = r.timestamp.strftime("%Y-%m-%d")
            by_day[day] = by_day.get(day, 0) + 1
        return StatsResponse(total=len(rows), by_type=by_type, by_day=by_day)
    finally:
        db.close()

@app.websocket("/ws/video")
async def ws_video(ws: WebSocket):
    await ws.accept()
    db = SessionLocal()
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            # Expected JSON: { "frame": "data:image/jpeg;base64,..." }
            frame_b64 = data.get("frame")
            if not frame_b64:
                continue
            frame_bgr, (w, h) = engine.decode_frame(frame_b64)
            # Ensure fixed frame size for recorder consistency
            frame_bgr = cv2.resize(frame_bgr, (640, 480))
            # Run inference
            result = engine.run(frame_bgr)
            # Draw overlays for preview
            overlay = engine.draw_overlays(frame_bgr, result)
            # Update recorder
            recorder.push_frame(overlay)

            snapshot_path = None
            video_path = None
            if result.get("alert", False) and cfg.get("recording_enabled", True):
                if recorder.active_writer is None:
                    video_path = recorder.trigger()
                else:
                    recorder.extend()

            # Persist events: store only on alerts
            if result.get("alert", False):
                # Save snapshot
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
                os.makedirs("snapshots", exist_ok=True)
                snapshot_path = os.path.join("snapshots", f"snapshot_{ts}.jpg")
                cv2.imwrite(snapshot_path, overlay)

                # Store each alert type as separate event (violence/fire)
                for et in result.get("alert_types", []):
                    conf = 0.0
                    if et == "violence":
                        conf = float(result.get("violence_score") or 0.0)
                    elif et == "fire":
                        # take max box confidence
                        boxes = result.get("fire_boxes") or []
                        conf = float(max([b["confidence"] for b in boxes], default=0.0))
                    ev = DetectionEvent(
                        event_type=et,
                        confidence=conf,
                        timestamp=datetime.utcnow(),
                        snapshot_path=snapshot_path,
                        video_path=recorder.active_path
                    )
                    db.add(ev)
                db.commit()

            # Encode overlay back to base64 for UI preview
            preview = engine.encode_jpeg(overlay, 80)
            out = {
                "preview": preview,
                "violence_score": result.get("violence_score"),
                "fire_boxes": result.get("fire_boxes"),
                "alert": result.get("alert", False),
                "alert_types": result.get("alert_types", []),
                "snapshot": snapshot_path,
                "video": recorder.active_path
            }
            await ws.send_text(json.dumps(out))
    except WebSocketDisconnect:
        pass
    finally:
        db.close()