import os
from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np
try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

class ViolenceModel:
    def __init__(self, model_path: str, sequence_length: int = 16, threshold: float = 0.6):
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.buffer: Deque[np.ndarray] = deque(maxlen=sequence_length)

        self.model = None
        self.model_path = model_path
        if tf is not None and os.path.isfile(model_path):
            # Load Keras .keras or H5 model
            self.model = tf.keras.models.load_model(model_path)

    def preprocess(self, frame_bgr: np.ndarray, size=(224, 224)) -> np.ndarray:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        img = img.astype(np.float32) / 127.5 - 1.0  # MobileNetV2 preprocess: [-1, 1]
        return img

    def update_and_predict(self, frame_bgr: np.ndarray) -> Optional[float]:
        self.buffer.append(self.preprocess(frame_bgr))
        if len(self.buffer) < self.sequence_length:
            return None
        if self.model is None or tf is None:
            # TensorFlow not available; skip prediction
            return None
        seq = np.stack(list(self.buffer), axis=0)  # (T, H, W, 3)
        seq = np.expand_dims(seq, axis=0)          # (1, T, H, W, 3)
        # Model expected input: adjust if your model expects channels first or a different shape
        preds = self.model.predict(seq, verbose=0)
        # Assume output is violence probability at index 0
        score = float(preds.squeeze())
        return score

    def is_alert(self, score: Optional[float]) -> bool:
        if score is None:
            return False
        return score >= self.threshold