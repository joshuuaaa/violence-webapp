import os
import cv2
import time
from collections import deque
from datetime import datetime
from typing import Deque, Optional

class IncidentRecorder:
    def __init__(self, output_dir="recordings", fps=5, frame_size=(640, 480), pre_roll=3, post_roll=5):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.fps = fps
        self.frame_size = frame_size
        self.pre_roll = pre_roll
        self.post_roll = post_roll

        self.buffer: Deque = deque(maxlen=pre_roll * fps)
        self.active_writer: Optional[cv2.VideoWriter] = None
        self.active_path: Optional[str] = None
        self.post_frames_remaining = 0

    def push_frame(self, frame_bgr):
        # keep rolling buffer for pre-roll
        self.buffer.append(frame_bgr)

        # if actively recording, write
        if self.active_writer is not None:
            self.active_writer.write(frame_bgr)
            if self.post_frames_remaining > 0:
                self.post_frames_remaining -= 1
                if self.post_frames_remaining == 0:
                    self._stop_recording()

    def trigger(self) -> str:
        # Start a new recording if not currently recording
        if self.active_writer is not None:
            # extend current recording post-roll
            self.post_frames_remaining = self.post_roll * self.fps
            return self.active_path or ""

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"incident_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.fps, self.frame_size)

        # write pre-roll buffer first
        for f in list(self.buffer):
            writer.write(f)

        self.active_writer = writer
        self.active_path = path
        self.post_frames_remaining = self.post_roll * self.fps
        return path

    def extend(self):
        # Reset post-roll countdown when another alert is detected
        if self.active_writer is not None:
            self.post_frames_remaining = self.post_roll * self.fps

    def _stop_recording(self):
        if self.active_writer is not None:
            self.active_writer.release()
        self.active_writer = None
        self.active_path = None
        self.post_frames_remaining = 0