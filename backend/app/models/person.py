import os
from typing import Optional, List, Dict

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None


class PersonDetector:
    def __init__(self, conf: float = 0.35, device: str = "", model_name: str = "yolov5n", infer_size: int = 416):
        """
        model_name: one of yolov5n|yolov5s|yolov5m (default: yolov5n for speed)
        infer_size: inference image size (default 416 for lower latency)
        """
        self.conf = conf
        self.model = None
        self.model_name = model_name or "yolov5n"
        self.infer_size = int(infer_size) if infer_size else 416
        if torch is None:
            print("[PersonDetector] PyTorch not available; person gate disabled.")
            return
        try:
            # Load general COCO model from cache/GitHub (default yolov5n for speed)
            self.model = torch.hub.load("ultralytics/yolov5", self.model_name, trust_repo=True)
            if device:
                self.model.to(device)
            self.model.conf = float(conf)
            # Reduce max detections to speed up NMS
            try:
                self.model.max_det = 50
            except Exception:
                pass
            print(f"[PersonDetector] loaded {self.model_name} (conf={self.model.conf}, size={self.infer_size})")
        except Exception as e:
            print(f"[PersonDetector] load error: {e}")
            self.model = None

    def detect(self, frame_bgr) -> Optional[List[Dict]]:
        if self.model is None:
            return None
        try:
            # Run at a smaller inference size to reduce latency
            res = self.model(frame_bgr, size=self.infer_size)
            # class 0 is 'person' in COCO for YOLOv5
            det = res.xyxy[0] if hasattr(res, "xyxy") and len(res.xyxy) else None
            if det is None:
                return []
            out: List[Dict] = []
            for *xyxy, conf, cls in det.cpu().numpy():
                if int(cls) == 0 and float(conf) >= self.model.conf:
                    x1, y1, x2, y2 = [int(float(v)) for v in xyxy]
                    out.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "confidence": float(conf)
                    })
            print(f"[PersonDetector] detect -> {len(out)} persons")
            return out
        except Exception as e:
            print(f"[PersonDetector] inference error: {e}")
            return None

    def count(self, frame_bgr) -> Optional[int]:
        dets = self.detect(frame_bgr)
        if dets is None:
            return None
        return len(dets)
