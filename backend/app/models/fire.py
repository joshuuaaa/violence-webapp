import os
from typing import List
import numpy as np

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

class FireModel:
    def __init__(self, weights_path: str, threshold: float = 0.4, device: str = ""):
        self.threshold = threshold
        self.model = None
        self.names = None
        self.weights_path = weights_path
        if torch is None:
            print("[FireModel] PyTorch not available; fire detection disabled.")
            return
        if not os.path.isfile(weights_path):
            print(f"[FireModel] Weights file not found at {weights_path}; fire detection disabled.")
            return
        # Use torch.hub to load YOLOv5 custom weights. Requires internet on first run.
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=weights_path, trust_repo=True)
        if device:
            self.model.to(device)
        self.model.conf = threshold  # confidence threshold
        self.names = self.model.names

    def predict(self, frame_bgr: np.ndarray):
        # YOLOv5 accepts BGR numpy directly
        if self.model is None:
            return []
        results = self.model(frame_bgr)  # inference
        # Parse results
        boxes = []
        # results.xyxy[0]: (N,6): x1,y1,x2,y2,conf,cls
        if hasattr(results, "xyxy") and len(results.xyxy):
            det_tensor = results.xyxy[0]
            det = det_tensor.detach().cpu().numpy() if hasattr(det_tensor, "detach") else det_tensor.numpy()
            names = self.names or {}
            for x1, y1, x2, y2, conf, cls in det:
                label = names[int(cls)]
                # Expect label 'fire' if model fine-tuned for fire detection
                boxes.append({
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    "label": label, "confidence": float(conf)
                })
        return boxes

    def is_alert(self, boxes: List[dict]) -> bool:
        return any(b["confidence"] >= self.threshold for b in boxes)