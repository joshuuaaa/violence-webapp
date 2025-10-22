import os
import asyncio
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
    "fps": 8,
    "enable_violence": True,
    "enable_fire": True,
    "violence_sequence_length": 16,  # MUST match model training (16 frames)
    "violence_threshold": 0.80,
    "violence_ema_alpha": 0.7,
    "violence_sustain_frames": 6,
    "violence_release_frames": 3,
    "violence_motion_min_ratio": 0.04,
    # Violence output interpretation knobs
    "violence_output_mode": "auto",  # auto | sigmoid | softmax
    "violence_softmax_index": 1,
    "violence_invert_score": False,
    "violence_temperature": 1.0,
    "violence_auto_invert": True,
    "violence_auto_invert_warmup": 32,
    "violence_auto_invert_high": 0.8,
    "violence_auto_invert_low": 0.2,
    # Person gate (optional): require at least N persons for violence alert
    "enable_person_gate": True,
    "person_require_proximity": False,
    "person_conf_threshold": 0.35,
    "person_every_n": 3,
    "person_model_name": "yolov5s",
    "person_infer_size": 512,
    "violence_min_persons": 2,
    "person_ttl_seconds": 1.0,
    "person_max_pair_distance_ratio": 0.15,
    "person_min_pair_iou": 0.12,
    "fire_threshold": 0.62,
    "fire_min_area": 1000,
    "fire_sustain_frames": 4,
    "fire_release_frames": 2,
    "fire_allowed_labels": ["fire"],
    "fire_backend": "yolov5",
    "fire_color_gate_enabled": True,
    "fire_color_min_ratio": 0.20,
    "fire_color_min_ratio_small": 0.14,
    "fire_color_small_area": 900,
    "fire_iou_min": 0.40,
    "fire_motion_min_ratio": 0.03,
    "fire_infer_size": 640,
    "fire_every_n": 3,
    "violence_model_path": "models/violence_model.keras",
    # Secondary violence (optional ensemble/confirm)
    "enable_violence_secondary": False,
    "violence_secondary_backend": "onnx",
    "violence_secondary_model_path": "models/violence_secondary.onnx",
    "violence_secondary_T": 16,
    "violence_secondary_input_size": 224,
    "violence_fusion_mode": "confirm",
    "violence_fusion_weight_secondary": 0.5,
    "violence_secondary_threshold": 0.7,
    "fire_model_weights": "models/best.pt",
    "recording_enabled": False,  # Disable recording by default for testing
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
    print("[WebSocket] Client connected")
    try:
        while True:
            try:
                # Receive at least one message, then quickly drain to get the most recent frame
                msg = await ws.receive_text()
                # Drain backlog to avoid processing stale frames when inference is slow
                while True:
                    try:
                        nxt = await asyncio.wait_for(ws.receive_text(), timeout=0.0001)
                        msg = nxt
                    except asyncio.TimeoutError:
                        break
                    except WebSocketDisconnect:
                        raise
                data = json.loads(msg)
                # Keepalive: respond to ping
                if isinstance(data, dict) and data.get("type") == "ping":
                    try:
                        await ws.send_text(json.dumps({"type": "pong"}))
                    except Exception:
                        pass
                    continue
                # Expected JSON: { "frame": "data:image/jpeg;base64,..." }
                frame_b64 = data.get("frame")
                if not frame_b64:
                    # Ignore non-frame messages quietly
                    continue
                frame_bgr, (w, h) = engine.decode_frame(frame_b64)
            except WebSocketDisconnect:
                print("[WebSocket] Client disconnected during receive")
                break
            except json.JSONDecodeError as e:
                print(f"[WebSocket] JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"[WebSocket] Frame decode error: {e}")
                continue
            # Ensure fixed frame size for recorder consistency
            frame_bgr = cv2.resize(frame_bgr, (640, 480))
            # Run inference (guard against occasional model errors)
            try:
                result = engine.run(frame_bgr)
            except Exception as e:
                # Don't crash the websocket loop on inference errors.
                # Send a fallback preview so the UI doesn't freeze on a stale frame.
                print(f"[WebSocket] Inference error: {e}")
                overlay = frame_bgr
                preview = engine.encode_jpeg(overlay, 80)
                out = {
                    "preview": preview,
                    "violence_score": None,
                    "fire_boxes": [],
                    "alert": False,
                    "alert_types": [],
                    "snapshot": None,
                    "video": recorder.active_path
                }
                try:
                    await ws.send_text(json.dumps(out))
                except Exception as se:
                    print(f"[WebSocket] Send error after inference failure: {se}")
                    break
                continue
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
            try:
                await ws.send_text(json.dumps(out))
            except Exception as e:
                print(f"[WebSocket] Send error: {e}")
                break
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected normally")
    except Exception as e:
        print(f"[WebSocket] Unexpected error: {e}")
    finally:
        print("[WebSocket] Cleaning up connection")
        db.close()