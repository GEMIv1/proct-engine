from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CAPTURE_DIR = _PROJECT_ROOT / "reports" / "violation_captures"

_JPEG_QUALITY = 85


class ScreenshotCapture:
    def __init__(self, capture_dir: str | Path | None = None, cooldown: float = 10.0, submission_id: str | None = None) -> None:
        self.capture_dir = Path(capture_dir) if capture_dir else _DEFAULT_CAPTURE_DIR
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown = cooldown
        self.submission_id = submission_id
        self._last_capture: dict[str, float] = {}

    def capture(self, frame: np.ndarray, violation_type: str, timestamp: str | None = None) -> str | None:
        if not self._can_capture(violation_type):
            return None

        self._last_capture[violation_type] = time.time()

        safe_ts = self._safe_timestamp(timestamp)
        short_id = uuid.uuid4().hex[:6]
        if self.submission_id:
            filename = f"{self.submission_id}_{violation_type}_{safe_ts}_{short_id}.jpg"
        else:
            filename = f"{violation_type}_{safe_ts}_{short_id}.jpg"
        full_path = self.capture_dir / filename

        cv2.imwrite(
            str(full_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
        )

        relative = f"violation_captures/{filename}"
        return relative

    def _can_capture(self, violation_type: str) -> bool:
        if self.cooldown <= 0:
            return True
        last = self._last_capture.get(violation_type, 0.0)
        return (time.time() - last) >= self.cooldown

    @staticmethod
    def _safe_timestamp(ts: str | None) -> str:
        if ts is None:
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            return ts

        return (
            ts.replace(":", "")
              .replace("-", "")
              .replace("T", "_")
              .replace("+", "_")
              .replace(".", "_")
              .replace(" ", "_")
        )
