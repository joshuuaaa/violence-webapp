import os
from collections import deque
from typing import Deque, Optional, List

import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None


class SecondaryViolenceModel:
    """
    Optional secondary violence scorer (e.g., ONNX), used to re-rank or confirm primary model.

    Expected ONNX input (default assumption):
      - shape: (1, T, H, W, 3) float32 in range [-1, 1] (MobileNet-style). Adjust if needed.
      - output: (1, 1) or (1,) score in [0,1] for violence probability.

    This is intentionally generic; adapt preprocess or dims to your exported model if different.
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "onnx",
        T: int = 16,
        input_size: int = 224,
    ):
        self.model_path = model_path
        self.backend = (backend or "onnx").lower()
        self.T = int(T)
        self.input_size = int(input_size)
        self.session = None
        self.input_name = None
        self.output_name = None
        if self.backend == "onnx":
            if ort is None:
                print("[SecondaryViolence] onnxruntime not available; secondary disabled.")
            elif not os.path.isfile(model_path):
                print(f"[SecondaryViolence] Model not found at {model_path}; secondary disabled.")
            else:
                try:
                    providers = [
                        ("CUDAExecutionProvider", {}),
                        ("CPUExecutionProvider", {}),
                    ]
                    self.session = ort.InferenceSession(model_path, providers=[p[0] for p in providers if p[0] in (ort.get_available_providers() if ort else [])] or ["CPUExecutionProvider"])  # type: ignore
                    ins = self.session.get_inputs()
                    outs = self.session.get_outputs()
                    self.input_name = ins[0].name if ins else None
                    self.output_name = outs[0].name if outs else None
                    print(f"[SecondaryViolence] ONNX loaded: {model_path} in={self.input_name} out={self.output_name}")
                except Exception as e:
                    print(f"[SecondaryViolence] ONNX load error: {e}")
                    self.session = None
        else:
            print(f"[SecondaryViolence] Unsupported backend '{self.backend}'. Only 'onnx' is implemented.")

    def available(self) -> bool:
        return self.session is not None

    def predict_from_buffer(self, buf: Deque[np.ndarray]) -> Optional[float]:
        """
        Expects buf of preprocessed frames shaped (H,W,3) in [-1,1] RGB, like primary model uses.
        Will pad/trim to T.
        """
        if not self.available():
            return None
        try:
            frames = list(buf)
            if len(frames) == 0:
                return None
            # Resize/pad/trim to T
            if len(frames) >= self.T:
                frames = frames[-self.T:]
            else:
                last = frames[-1]
                frames = frames + [last] * (self.T - len(frames))
            seq = np.stack(frames, axis=0).astype(np.float32)  # (T,H,W,3) already [-1,1]
            seq = np.expand_dims(seq, axis=0)  # (1,T,H,W,3)
            inp = {self.input_name: seq} if self.input_name else {self.session.get_inputs()[0].name: seq}  # type: ignore
            out = self.session.run([self.output_name] if self.output_name else None, inp)  # type: ignore
            y = out[0]
            score = float(np.squeeze(y))
            # clamp just in case
            if np.isnan(score):
                return None
            score = max(0.0, min(1.0, score))
            return score
        except Exception as e:
            print(f"[SecondaryViolence] predict error: {e}")
            return None
