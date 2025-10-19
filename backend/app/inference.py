import os
import cv2
import base64
import numpy as np
from typing import Tuple, Dict, Any
from .models.violence import ViolenceModel
from .models.fire import FireModel

class InferenceEngine:
    def __init__(self, config: dict):
        self.config = config
        self.violence_model = None
        self.fire_model = None

        if self.config.get("enable_violence", True):
            v_path = self.config.get("violence_model_path") or os.path.join("models", "violence_model.keras")
            self.violence_model = ViolenceModel(
                model_path=v_path,
                sequence_length=self.config.get("violence_sequence_length", 16),
                threshold=float(self.config.get("violence_threshold", 0.6)),
            )
        if self.config.get("enable_fire", True):
            f_weights = self.config.get("fire_model_weights") or os.path.join("models", "best.pt")
            self.fire_model = FireModel(
                weights_path=f_weights,
                threshold=float(self.config.get("fire_threshold", 0.4)),
            )

    @staticmethod
    def decode_frame(data_url: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        # data_url like "data:image/jpeg;base64,...."
        comma = data_url.find(",")
        b64 = data_url[comma+1:] if comma != -1 else data_url
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        h, w = frame.shape[:2]
        return frame, (w, h)

    def run(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        result = {
            "violence_score": None,
            "fire_boxes": [],
            "alert": False,
            "alert_types": []
        }

        # Violence
        if self.violence_model is not None:
            v_score = self.violence_model.update_and_predict(frame_bgr)
            result["violence_score"] = v_score
            if self.violence_model.is_alert(v_score):
                result["alert"] = True
                result["alert_types"].append("violence")

        # Fire
        if self.fire_model is not None:
            boxes = self.fire_model.predict(frame_bgr)
            result["fire_boxes"] = boxes
            if self.fire_model.is_alert(boxes):
                result["alert"] = True
                result["alert_types"].append("fire")

        return result

    @staticmethod
    def draw_overlays(frame_bgr: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        out = frame_bgr.copy()
        # Draw fire boxes
        for b in result.get("fire_boxes", []):
            color = (0, 0, 255) if b["label"].lower().startswith("fire") else (255, 0, 0)
            cv2.rectangle(out, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, 2)
            txt = f"{b['label']} {b['confidence']:.2f}"
            cv2.putText(out, txt, (b["x1"], max(b["y1"] - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw violence score
        v = result.get("violence_score")
        if v is not None:
            color = (0, 255, 255) if v < 0.5 else (0, 0, 255)
            cv2.putText(out, f"Violence: {v:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return out

    @staticmethod
    def encode_jpeg(frame_bgr: np.ndarray, quality: int = 80) -> str:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ok, buf = cv2.imencode(".jpg", frame_bgr, encode_param)
        if not ok:
            return ""
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"