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
    def __init__(self, model_path: str, sequence_length: int = 16, threshold: float = 0.6, pad_short_sequences: bool = True,
                 ema_alpha: float = 0.6, sustain_frames: int = 3, release_frames: int = 3,
                 output_mode: str = "auto", softmax_index: int = 1, invert_score: bool = False, temperature: float = 1.0,
                 auto_invert: bool = True, auto_invert_warmup: int = 32, auto_invert_high: float = 0.8, auto_invert_low: float = 0.2):
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.pad_short_sequences = pad_short_sequences
        self.ema_alpha = ema_alpha
        self.sustain_frames = sustain_frames
        self.release_frames = release_frames
    # Sliding buffer of preprocessed frames
        self.buffer = deque(maxlen=sequence_length)
        self.model_T = None  # model-declared temporal length if known
        self.last_score = None
        self.smoothed_score = None
        self.above_count = 0
        self.below_count = 0
        # Output interpretation
        self.output_mode = (output_mode or "auto").lower()
        self.softmax_index = int(softmax_index)
        self.invert_score = bool(invert_score)
        self.temperature = float(temperature) if temperature else 1.0
        # Auto-invert heuristic (for 1-unit heads where semantics might be inverted)
        self.auto_invert = bool(auto_invert)
        self.auto_invert_warmup = int(auto_invert_warmup)
        self.auto_invert_high = float(auto_invert_high)
        self.auto_invert_low = float(auto_invert_low)
        self._runtime_invert_decided = False
        self._runtime_invert = False
        self._warmup_scores = deque(maxlen=max(1, self.auto_invert_warmup))
        # Debug/telemetry fields
        self.last_raw = None
        self.last_raw_shape = None
        self.last_prob_raw = None  # before EMA
        self.last_prob = None      # after EMA

        self.model = None
        self.model_path = model_path
        if tf is not None and os.path.isfile(model_path):
            # Load Keras .keras or H5 model
            self.model = tf.keras.models.load_model(model_path)
            # Try to infer expected temporal length (T) from model's input shape
            try:
                # Prefer model.inputs[0].shape if available
                t_dim = None
                try:
                    if hasattr(self.model, "inputs") and self.model.inputs:
                        shp = self.model.inputs[0].shape
                        # shp is a tf.TensorShape, index 1 is time dimension
                        t_candidate = shp[1]
                        if t_candidate is not None and int(t_candidate) > 0:
                            t_dim = int(t_candidate)
                except Exception:
                    pass
                if t_dim is None:
                    in_shape = getattr(self.model, "input_shape", None)
                    if isinstance(in_shape, list) and in_shape:
                        in_shape = in_shape[0]
                    if in_shape and len(in_shape) >= 2:
                        try:
                            t_candidate = in_shape[1]
                            if t_candidate is not None and int(t_candidate) > 0:
                                t_dim = int(t_candidate)
                        except Exception:
                            t_dim = None
                if t_dim is not None:
                    self.model_T = t_dim
                    if t_dim != self.sequence_length:
                        print(f"[ViolenceModel] Adjusting sequence_length to model T={t_dim}")
                        self.sequence_length = t_dim
                        self.buffer = deque(maxlen=self.sequence_length)
            except Exception as e:
                print(f"[ViolenceModel] Could not infer sequence length from model: {e}")

    def preprocess(self, frame_bgr: np.ndarray, size=(224, 224)) -> np.ndarray:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        img = img.astype(np.float32) / 127.5 - 1.0  # MobileNetV2 preprocess: [-1, 1]
        return img

    def update_and_predict(self, frame_bgr: np.ndarray) -> Optional[float]:
        # Update sliding window buffer
        self.buffer.append(self.preprocess(frame_bgr))
        # Determine effective T expected by model
        effective_T = self.model_T or self.sequence_length
        # If not enough frames yet, optionally pad to allow early predictions
        if len(self.buffer) < effective_T and not self.pad_short_sequences:
            return None
        # If model unavailable, return last known score (or None)
        if self.model is None or tf is None:
            return self.last_score
        # Prepare sequence: slice last T frames; pad if needed
        buf_list = list(self.buffer)
        if len(buf_list) >= effective_T:
            buf_list = buf_list[-effective_T:]
        else:
            # pad with last available frame to reach T
            if self.pad_short_sequences and len(buf_list) > 0:
                last = buf_list[-1]
                pad_count = effective_T - len(buf_list)
                buf_list = buf_list + [last] * pad_count
        seq = np.stack(buf_list, axis=0)  # (T, H, W, 3)
        seq = np.expand_dims(seq, axis=0)  # (1, T, H, W, 3)
        try:
            preds = self.model.predict(seq, verbose=0)
            # Interpret model outputs flexibly
            arr = np.array(preds)
            arr = np.squeeze(arr)
            # Save raw for debugging
            try:
                self.last_raw = arr
                self.last_raw_shape = getattr(arr, "shape", None)
            except Exception:
                self.last_raw = None
                self.last_raw_shape = None
            score: Optional[float] = None
            # Optional temperature scaling on logits, if identifiable
            def _sigmoid(x):
                return 1.0 / (1.0 + np.exp(-x))
            mode = self.output_mode
            try:
                if mode == "auto":
                    if arr.shape == ():
                        # scalar -> already a probability/logit; try to bound
                        s = float(arr)
                        # Heuristic: if outside [0,1], treat as logit
                        if s < 0.0 or s > 1.0:
                            s = float(_sigmoid(s / self.temperature))
                        score = s
                    elif isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.size == 2:
                        # two-class output: pick index (default 1)
                        idx = max(0, min(1, self.softmax_index))
                        # If values don't sum to ~1, apply softmax
                        v = arr.astype(np.float32)
                        if not (0.99 <= float(np.sum(v)) <= 1.01 and np.all(v >= 0)):
                            # apply temperature softmax
                            vv = v / max(1e-6, self.temperature)
                            vv = vv - np.max(vv)
                            expv = np.exp(vv)
                            v = expv / np.sum(expv)
                        score = float(v[idx])
                    else:
                        # Fallback: take mean bounded to [0,1]
                        m = float(np.mean(arr))
                        score = min(1.0, max(0.0, m))
                elif mode == "sigmoid":
                    s = float(arr if arr.shape == () else np.mean(arr))
                    if s < 0.0 or s > 1.0:
                        s = float(_sigmoid(s / self.temperature))
                    score = s
                elif mode == "softmax":
                    idx = int(self.softmax_index)
                    v = arr.astype(np.float32)
                    if v.ndim == 0:
                        score = float(min(1.0, max(0.0, v)))
                    else:
                        vv = v / max(1e-6, self.temperature)
                        vv = vv - np.max(vv)
                        expv = np.exp(vv)
                        prob = expv / np.sum(expv)
                        idx = max(0, min(len(prob) - 1, idx))
                        score = float(prob[idx])
                else:
                    score = float(np.mean(arr))
            except Exception:
                score = float(np.mean(arr))

            # At this point, 'score' is the probability from raw outputs before any inversion/EMA
            # Record raw probability (pre-EMA) for visibility and warmup
            if score is not None:
                self.last_prob_raw = float(score)
                if self.auto_invert and not self._runtime_invert_decided:
                    self._warmup_scores.append(float(score))
                    if len(self._warmup_scores) >= self.auto_invert_warmup:
                        try:
                            med = float(np.median(self._warmup_scores))
                            if med >= self.auto_invert_high:
                                self._runtime_invert = True
                                print(f"[ViolenceModel] Auto-invert enabled (median={med:.3f} over {len(self._warmup_scores)} frames)")
                            elif med <= self.auto_invert_low:
                                self._runtime_invert = False
                                print(f"[ViolenceModel] Auto-invert disabled (median={med:.3f})")
                            self._runtime_invert_decided = True
                        except Exception:
                            self._runtime_invert_decided = True
            # Apply explicit or runtime inversion prior to EMA
            do_invert = bool(self.invert_score or (self.auto_invert and self._runtime_invert))
            if do_invert and score is not None:
                score = 1.0 - score
            # Apply EMA smoothing
            if self.smoothed_score is None:
                self.smoothed_score = score
            else:
                a = float(self.ema_alpha)
                self.smoothed_score = a * score + (1 - a) * self.smoothed_score
            score = float(self.smoothed_score)
            # Store smoothed prob for UI/debug
            self.last_prob = float(score)
            self.last_score = score
            return score
        except Exception as e:
            # Swallow intermittent TF predict issues and reuse last score
            print(f"[ViolenceModel] predict error: {e}")
            # Fallback: try with last 16 frames if available
            try:
                if len(self.buffer) >= 16:
                    seq16 = np.stack(list(self.buffer)[-16:], axis=0)
                    seq16 = np.expand_dims(seq16, axis=0)
                    preds = self.model.predict(seq16, verbose=0)
                    score = float(np.squeeze(preds))
                    self.last_score = score
                    return score
            except Exception as e2:
                print(f"[ViolenceModel] fallback(16) predict error: {e2}")
            return self.last_score

    def is_alert(self, score: Optional[float]) -> bool:
        if score is None:
            return False
        # Hysteresis with sustain/release frames
        if score >= self.threshold:
            self.above_count += 1
            self.below_count = 0
        else:
            self.below_count += 1
            self.above_count = 0
        if self.above_count >= self.sustain_frames:
            return True
        if self.below_count >= self.release_frames:
            return False
        # Default to previous state if available
        return (self.above_count > 0 and self.below_count == 0)