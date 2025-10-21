import os
from typing import List
import numpy as np
import cv2

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

class FireModel:
    def __init__(
        self,
        weights_path: str,
        threshold: float = 0.4,
        device: str = "",
        min_area: int = 800,
        sustain_frames: int = 2,
        release_frames: int = 2,
        allowed_labels: list[str] | None = None,
        color_gate_enabled: bool = True,
        color_min_ratio: float = 0.25,
        iou_min: float = 0.3,
        backend: str = "yolov5",
        motion_min_ratio: float = 0.0,
        color_min_ratio_small: float = 0.12,
        color_small_area: int = 800,
        infer_size: int = 512,
    ):
        self.threshold = threshold
        self.min_area = int(min_area)
        self.sustain_frames = int(sustain_frames)
        self.release_frames = int(release_frames)
        self.allowed_labels = [l.lower() for l in (allowed_labels or ["fire"])]
        self.above_count = 0
        self.below_count = 0
        self.model = None
        self.names = None
        self.weights_path = weights_path
        self.prev_boxes = []  # for simple temporal consistency via IoU
        # Color/IoU gates
        self.color_gate_enabled = bool(color_gate_enabled)
        self.color_min_ratio = float(color_min_ratio)
        self.color_min_ratio_small = float(color_min_ratio_small)
        self.color_small_area = int(color_small_area)
        self.iou_min = float(iou_min)
        # Debug cadence counter
        self._dbg_seen = 0
        # Backend selection
        self.backend = (backend or "yolov5").lower()
        # Optional motion gate inside detected boxes
        self.motion_min_ratio = float(motion_min_ratio)
        self._prev_gray = None
        # Inference size for YOLOv5
        self.infer_size = int(infer_size) if infer_size else 512

        if self.backend == "yolov5":
            if torch is None:
                print("[FireModel] PyTorch not available; YOLOv5 backend disabled.")
            elif not os.path.isfile(weights_path):
                print(f"[FireModel] Weights file not found at {weights_path}; YOLOv5 backend disabled.")
            else:
                # Use torch.hub to load YOLOv5 custom weights. Requires internet on first run.
                self.model = torch.hub.load("ultralytics/yolov5", "custom", path=weights_path, trust_repo=True)
                if device:
                    self.model.to(device)
                self.model.conf = threshold  # confidence threshold
                self.names = self.model.names
                try:
                    print(f"[FireModel] backend=yolov5 weights={weights_path} classes={list(self.names.values()) if isinstance(self.names, dict) else self.names}")
                except Exception:
                    pass
        elif self.backend == "heuristic":
            print("[FireModel] Using heuristic fire detector (HSV + contours)")
        else:
            print(f"[FireModel] Unknown backend '{self.backend}', falling back to heuristic")
            self.backend = "heuristic"

    def predict(self, frame_bgr: np.ndarray):
        # prepare grayscale for optional motion gating
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = None
        boxes = []
        if self.backend == "yolov5":
            # YOLOv5 inference path
            if self.model is None:
                return []
            try:
                # Run YOLOv5 at a reduced size for lower latency (configurable)
                results = self.model(frame_bgr, size=self.infer_size)
            except Exception as e:
                print(f"[FireModel] inference error: {e}")
                return []
            raw_count = 0
            raw_labels = []
            top_conf = 0.0
            if hasattr(results, "xyxy") and len(results.xyxy):
                det_tensor = results.xyxy[0]
                det = det_tensor.detach().cpu().numpy() if hasattr(det_tensor, "detach") else det_tensor.numpy()
                names = self.names or {}
                for x1, y1, x2, y2, conf, cls in det:
                    raw_count += 1
                    label = str(names[int(cls)]) if int(cls) in names else str(int(cls))
                    label_l = label.lower()
                    raw_labels.append(label_l)
                    if float(conf) > top_conf:
                        top_conf = float(conf)
                    area = max(0, int(x2 - x1)) * max(0, int(y2 - y1))
                    if label_l in self.allowed_labels and area >= self.min_area:
                        boxes.append({
                            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                            "label": label, "confidence": float(conf), "area": area
                        })
            # Debug
            self._dbg_seen = (self._dbg_seen + 1) % 60
            if (self._dbg_seen == 0) or (raw_count > 0 and not boxes):
                print(f"[FireModel] raw={raw_count} labels={set(raw_labels)} top_conf={top_conf:.2f} after label/area={len(boxes)} (min_area={self.min_area}, allowed={self.allowed_labels})")
            # Optional color gate
            try:
                filtered = []
                if boxes:
                    min_ratio = self.color_min_ratio
                    enable = self.color_gate_enabled
                    for b in boxes:
                        if not enable:
                            filtered.append(b)
                            continue
                        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
                        roi = frame_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                        if roi.size == 0:
                            continue
                        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        lower1 = np.array([0, 80, 120]); upper1 = np.array([25, 255, 255])
                        lower2 = np.array([160, 80, 120]); upper2 = np.array([179, 255, 255])
                        mask1 = cv2.inRange(hsv, lower1, upper1)
                        mask2 = cv2.inRange(hsv, lower2, upper2)
                        warm = cv2.bitwise_or(mask1, mask2)
                        ratio = float(np.count_nonzero(warm)) / float(warm.size)
                        # adapt color threshold for small boxes
                        area = (x2 - x1) * (y2 - y1)
                        min_r = min_ratio if area >= self.color_small_area else self.color_min_ratio_small
                        if ratio >= min_r:
                            filtered.append(b)
                    if self._dbg_seen == 0 and self.color_gate_enabled:
                        print(f"[FireModel] after color gate={len(filtered)} (min_ratio={min_ratio}, small_min_ratio={self.color_min_ratio_small}, small_area<{self.color_small_area})")
                else:
                    filtered = boxes
                boxes = filtered
            except Exception:
                pass
        else:
            # Heuristic HSV-based fire detection
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            lower1 = np.array([0, 80, 120]); upper1 = np.array([25, 255, 255])
            lower2 = np.array([160, 80, 120]); upper2 = np.array([179, 255, 255])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            warm = cv2.bitwise_or(mask1, mask2)
            # remove noise
            kernel = np.ones((5,5), np.uint8)
            warm = cv2.morphologyEx(warm, cv2.MORPH_OPEN, kernel, iterations=1)
            warm = cv2.dilate(warm, kernel, iterations=1)
            contours, _ = cv2.findContours(warm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = int(w * h)
                if area < self.min_area:
                    continue
                roi = warm[y:y+h, x:x+w]
                ratio = float(np.count_nonzero(roi)) / float(roi.size)
                conf = float(min(1.0, ratio * 1.5))
                boxes.append({
                    "x1": int(x), "y1": int(y), "x2": int(x+w), "y2": int(y+h),
                    "label": "fire", "confidence": conf, "area": area
                })
        # Optional within-box motion gating (both backends)
        try:
            if self.motion_min_ratio > 0.0 and gray is not None and self._prev_gray is not None and boxes:
                kept = []
                for b in boxes:
                    x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = max(0, x2), max(0, y2)
                    roi_now = gray[y1c:y2c, x1c:x2c]
                    roi_prev = self._prev_gray[y1c:y2c, x1c:x2c]
                    if roi_now.size == 0 or roi_prev.size == 0 or roi_now.shape != roi_prev.shape:
                        continue
                    diff = cv2.absdiff(roi_now, roi_prev)
                    _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    ratio = float(np.count_nonzero(th)) / float(th.size)
                    if ratio >= self.motion_min_ratio:
                        kept.append(b)
                if self._dbg_seen == 0:
                    print(f"[FireModel] after motion gate kept={len(kept)} (min_ratio={self.motion_min_ratio})")
                boxes = kept
        except Exception:
            pass

        # IoU tracking (both backends)
        try:
            iou_min = self.iou_min
            if self.prev_boxes and boxes:
                kept = []
                for b in boxes:
                    bx = (b["x1"], b["y1"], b["x2"], b["y2"])
                    if any(self._iou(bx, (p["x1"], p["y1"], p["x2"], p["y2"])) >= iou_min for p in self.prev_boxes):
                        kept.append(b)
                boxes = kept
            self.prev_boxes = boxes
        except Exception:
            pass
        # update previous gray frame for next call
        try:
            if gray is not None:
                self._prev_gray = gray
        except Exception:
            pass
        return boxes

    def is_alert(self, boxes: List[dict]) -> bool:
        any_above = any(b["confidence"] >= self.threshold for b in boxes)
        if any_above:
            self.above_count += 1
            self.below_count = 0
        else:
            self.below_count += 1
            self.above_count = 0
        if self.above_count >= self.sustain_frames:
            return True
        if self.below_count >= self.release_frames:
            return False
        return self.above_count > 0 and self.below_count == 0

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = a_area + b_area - inter
        return 0.0 if union <= 0 else inter / union