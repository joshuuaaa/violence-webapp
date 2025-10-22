import os
import cv2
import base64
import numpy as np
from typing import Tuple, Dict, Any
from .models.violence import ViolenceModel
from .models.violence_secondary import SecondaryViolenceModel
from .models.fire import FireModel
from .models.person import PersonDetector

class InferenceEngine:
    def __init__(self, config: dict):
        self.config = config
        self.violence_model = None
        self.violence_secondary = None
        self.fire_model = None
        self.prev_gray = None  # for motion estimation
        self.person_detector = None
        self.person_frame_count = 0
        self._last_person_dets = None  # cache of last person detections (list of boxes)
        self._last_person_time = 0.0
        # Fire throttling/cache
        self._fire_frame_count = 0
        self._last_fire_boxes = []

        if self.config.get("enable_violence", True):
            v_path = self.config.get("violence_model_path") or os.path.join("models", "violence_model.keras")
            self.violence_model = ViolenceModel(
                model_path=v_path,
                sequence_length=int(self.config.get("violence_sequence_length", 16)),
                threshold=float(self.config.get("violence_threshold", 0.6)),
                ema_alpha=float(self.config.get("violence_ema_alpha", 0.6)),
                sustain_frames=int(self.config.get("violence_sustain_frames", 3)),
                release_frames=int(self.config.get("violence_release_frames", 3)),
                output_mode=str(self.config.get("violence_output_mode", "auto")),
                softmax_index=int(self.config.get("violence_softmax_index", 1)),
                invert_score=bool(self.config.get("violence_invert_score", False)),
                temperature=float(self.config.get("violence_temperature", 1.0)),
                auto_invert=bool(self.config.get("violence_auto_invert", True)),
                auto_invert_warmup=int(self.config.get("violence_auto_invert_warmup", 32)),
                auto_invert_high=float(self.config.get("violence_auto_invert_high", 0.8)),
                auto_invert_low=float(self.config.get("violence_auto_invert_low", 0.2)),
            )
            # Optional secondary for fusion/confirmation
            if bool(self.config.get("enable_violence_secondary", False)):
                try:
                    self.violence_secondary = SecondaryViolenceModel(
                        model_path=str(self.config.get("violence_secondary_model_path", "models/violence_secondary.onnx")),
                        backend=str(self.config.get("violence_secondary_backend", "onnx")),
                        T=int(self.config.get("violence_secondary_T", self.config.get("violence_sequence_length", 16))),
                        input_size=int(self.config.get("violence_secondary_input_size", 224)),
                    )
                    if not self.violence_secondary.available():
                        self.violence_secondary = None
                except Exception as e:
                    print(f"[Inference] Secondary violence init error: {e}")
                    self.violence_secondary = None
        if self.config.get("enable_fire", True):
            f_weights = self.config.get("fire_model_weights") or os.path.join("models", "best.pt")
            self.fire_model = FireModel(
                weights_path=f_weights,
                threshold=float(self.config.get("fire_threshold", 0.4)),
                min_area=int(self.config.get("fire_min_area", 800)),
                sustain_frames=int(self.config.get("fire_sustain_frames", 2)),
                release_frames=int(self.config.get("fire_release_frames", 2)),
                allowed_labels=self.config.get("fire_allowed_labels", ["fire"]),
                color_gate_enabled=bool(self.config.get("fire_color_gate_enabled", True)),
                color_min_ratio=float(self.config.get("fire_color_min_ratio", 0.25)),
                iou_min=float(self.config.get("fire_iou_min", 0.3)),
                backend=str(self.config.get("fire_backend", "yolov5")),
                motion_min_ratio=float(self.config.get("fire_motion_min_ratio", 0.0)),
                color_min_ratio_small=float(self.config.get("fire_color_min_ratio_small", 0.12)),
                color_small_area=int(self.config.get("fire_color_small_area", 800)),
                infer_size=int(self.config.get("fire_infer_size", 512)),
            )
        # Optional person detector gate
        if self.config.get("enable_person_gate", True):
            try:
                self.person_detector = PersonDetector(
                    conf=float(self.config.get("person_conf_threshold", 0.35)),
                    model_name=str(self.config.get("person_model_name", "yolov5n")),
                    infer_size=int(self.config.get("person_infer_size", 416))
                )
            except Exception as e:
                print(f"[Inference] Person detector init error: {e}")
                self.person_detector = None

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
        # ensure person_count is defined for later debug info
        person_count = None

        # Violence
        if self.violence_model is not None:
            v_score = self.violence_model.update_and_predict(frame_bgr)
            s_score = None
            # Optional secondary fusion/confirmation using primary's preprocessed buffer
            if self.violence_secondary is not None and hasattr(self.violence_model, "buffer"):
                try:
                    s_score = self.violence_secondary.predict_from_buffer(self.violence_model.buffer)
                except Exception as e:
                    print(f"[Inference] Secondary violence predict error: {e}")
                    s_score = None
            # Fuse scores before hysteresis
            fusion_mode = str(self.config.get("violence_fusion_mode", "confirm")).lower()
            w2 = float(self.config.get("violence_fusion_weight_secondary", 0.5))
            eff_score = v_score
            if s_score is not None and v_score is not None:
                if fusion_mode == "average":
                    w2 = max(0.0, min(1.0, w2))
                    eff_score = (1.0 - w2) * v_score + w2 * s_score
                elif fusion_mode == "confirm":
                    # Use primary score but gate alert on secondary confirmation later
                    eff_score = v_score
            result["violence_score"] = eff_score
            # Expose raw vs smoothed probability and raw output shape for debugging/overlay
            try:
                if hasattr(self.violence_model, "last_prob_raw") and self.violence_model.last_prob_raw is not None:
                    result["violence_prob_raw"] = float(self.violence_model.last_prob_raw)
                if hasattr(self.violence_model, "last_prob") and self.violence_model.last_prob is not None:
                    result["violence_prob_smoothed"] = float(self.violence_model.last_prob)
                if hasattr(self.violence_model, "last_raw_shape") and self.violence_model.last_raw_shape is not None:
                    # Convert numpy shape to a simple list for JSON
                    shp = self.violence_model.last_raw_shape
                    try:
                        result["violence_raw_shape"] = list(shp) if hasattr(shp, "__iter__") else shp
                    except Exception:
                        result["violence_raw_shape"] = str(shp)
            except Exception:
                pass
            if s_score is not None:
                result["violence_secondary_score"] = s_score
            # Motion gate: compute ratio of changed pixels using simple frame differencing
            motion_ratio = None
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                if self.prev_gray is not None:
                    diff = cv2.absdiff(gray, self.prev_gray)
                    _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    motion_ratio = float(np.count_nonzero(th)) / float(th.size)
                self.prev_gray = gray
            except Exception:
                pass
            motion_ok = True
            min_motion = float(self.config.get("violence_motion_min_ratio", 0.02))
            if motion_ratio is not None:
                motion_ok = motion_ratio >= min_motion
            result["motion_ok"] = motion_ok
            # Optional person gate: count every N frames with TTL cache and proximity
            gate_enabled = bool(self.config.get("enable_person_gate", True))
            persons_ok = not gate_enabled  # if gate disabled, allow by default; if enabled, block until proven
            person_gate_reason = None
            min_persons = int(self.config.get("violence_min_persons", 2))
            n = int(self.config.get("person_every_n", 3))
            if gate_enabled and self.person_detector is not None and n > 0:
                import time
                now = time.time()
                dets = None
                # use cache if within TTL
                ttl = float(self.config.get("person_ttl_seconds", 1.0))
                if self._last_person_dets is not None and (now - self._last_person_time) <= ttl:
                    dets = self._last_person_dets
                else:
                    self.person_frame_count = (self.person_frame_count + 1) % n
                    if self.person_frame_count == 0:
                        dets = self.person_detector.detect(frame_bgr)
                        if dets is not None:
                            self._last_person_dets = dets
                            self._last_person_time = now
                if dets is not None:
                    person_count = len(dets)
                    # persons_ok requires at least min_persons; for >= 2, require at least one close pair
                    persons_ok = person_count >= min_persons
                else:
                    # no recent detection -> keep gate closed
                    persons_ok = False
                    person_gate_reason = "no_recent_person_detection"
                # Optional proximity requirement (distance/IoU). Can be disabled via config.
                require_prox = bool(self.config.get("person_require_proximity", False))
                if require_prox and persons_ok and (person_count or 0) >= 2:
                    try:
                        h, w = frame_bgr.shape[:2]
                        diag = (w**2 + h**2) ** 0.5
                        max_ratio = float(self.config.get("person_max_pair_distance_ratio", 0.18))
                        min_iou = float(self.config.get("person_min_pair_iou", 0.05))
                        any_close = False
                        if dets and len(dets) >= 2:
                            for i in range(len(dets)):
                                for j in range(i+1, len(dets)):
                                    a, b = dets[i], dets[j]
                                    ax = (a["x1"] + a["x2"]) / 2.0
                                    ay = (a["y1"] + a["y2"]) / 2.0
                                    bx = (b["x1"] + b["x2"]) / 2.0
                                    by = (b["y1"] + b["y2"]) / 2.0
                                    dist = ((ax - bx)**2 + (ay - by)**2) ** 0.5
                                    # IoU
                                    x1 = max(a["x1"], b["x1"]) ; y1 = max(a["y1"], b["y1"]) ;
                                    x2 = min(a["x2"], b["x2"]) ; y2 = min(a["y2"], b["y2"]) ;
                                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                                    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) ;
                                    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) ;
                                    union = area_a + area_b - inter
                                    iou = (inter / union) if union > 0 else 0.0
                                    if (dist <= max_ratio * diag) or (iou >= min_iou):
                                        any_close = True
                                        break
                                if any_close:
                                    break
                        if not any_close:
                            persons_ok = False
                            person_gate_reason = "no_close_pair"
                    except Exception:
                        # if any computation fails, don't block
                        pass
            result["persons_ok"] = persons_ok
            # Final violence alert requires score + motion + persons (if enabled)
            # Apply hysteresis on effective score
            is_v = self.violence_model.is_alert(result.get("violence_score"))
            # Optional confirm gate using secondary
            if is_v and self.violence_secondary is not None and fusion_mode == "confirm":
                sec_thr = float(self.config.get("violence_secondary_threshold", 0.7))
                if (result.get("violence_secondary_score") or 0.0) < sec_thr:
                    is_v = False
                    person_gate_reason = (person_gate_reason or "") + "; no_secondary_confirm"
            if motion_ok and persons_ok and is_v:
                result["alert"] = True
                result["alert_types"].append("violence")
                print(f"[Inference] VIOLENCE alert: score={v_score:.3f} motion={motion_ok} persons={person_count} reason={person_gate_reason}")
            else:
                # log why we didn't alert for debugging
                print(f"[Inference] violence_check: score={v_score:.3f} is_v={is_v} motion={motion_ok} persons_ok={persons_ok} persons={person_count} reason={person_gate_reason}")

        # Fire
        if self.fire_model is not None:
            every_n = max(1, int(self.config.get("fire_every_n", 2)))
            if (self._fire_frame_count % every_n) == 0:
                boxes = self.fire_model.predict(frame_bgr)
                self._last_fire_boxes = boxes
            else:
                boxes = self._last_fire_boxes
            self._fire_frame_count = (self._fire_frame_count + 1) % 1000000
            result["fire_boxes"] = boxes
            if self.fire_model.is_alert(boxes):
                result["alert"] = True
                result["alert_types"].append("fire")

        # Include debug info for UI if needed
        try:
            if self.violence_model is not None:
                result["violence_T"] = self.violence_model.model_T or self.violence_model.sequence_length
                result["violence_buffer"] = len(self.violence_model.buffer)
                if person_count is not None:
                    result["persons"] = int(person_count)
                if self._last_person_dets is not None:
                    result["person_boxes"] = self._last_person_dets
                if person_gate_reason is not None:
                    result["person_gate_reason"] = person_gate_reason
        except Exception:
            pass
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
            # Optional tiny debug line for raw vs smoothed prob
            vr = result.get("violence_prob_raw")
            vs = result.get("violence_prob_smoothed")
            if vr is not None and vs is not None:
                cv2.putText(out, f"Raw/Sm: {vr:.2f}/{vs:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

        # Draw gate info and alert banner
        y = 70 if result.get("violence_prob_raw") is not None else 55
        if result.get("persons_ok") is not None:
            cv2.putText(out, f"Persons OK: {result.get('persons_ok')}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y += 20
        if result.get("motion_ok") is not None:
            cv2.putText(out, f"Motion OK: {result.get('motion_ok')}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y += 20
        if result.get("person_gate_reason"):
            cv2.putText(out, f"Gate: {result.get('person_gate_reason')}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
            y += 20
        if "violence" in (result.get("alert_types") or []):
            cv2.putText(out, "ALERT: VIOLENCE", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Draw person boxes (debug)
        for p in result.get("person_boxes", []) or []:
            cv2.rectangle(out, (p["x1"], p["y1"]), (p["x2"], p["y2"]), (0, 255, 255), 1)

        return out

    @staticmethod
    def encode_jpeg(frame_bgr: np.ndarray, quality: int = 80) -> str:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ok, buf = cv2.imencode(".jpg", frame_bgr, encode_param)
        if not ok:
            return ""
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"