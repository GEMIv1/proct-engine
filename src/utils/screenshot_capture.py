"""
screenshot_capture — saves annotated frames to disk when violations occur.

Each screenshot is written to ``reports/violation_captures/`` (or a custom
directory) as a JPEG file.  The filename encodes the violation type, a
human-readable timestamp, and a short random suffix to avoid collisions.

A per-violation-type cooldown prevents flooding the disk with near-identical
frames for sustained violations (e.g. prolonged gaze-away).

Usage::

    from utils.screenshot_capture import ScreenshotCapture
    capture = ScreenshotCapture()
    path = capture.capture(frame, "GAZE_AWAY", "2026-07-03T14:00:00+00:00")
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CAPTURE_DIR = _PROJECT_ROOT / "reports" / "violation_captures"

# JPEG quality (0–100). 85 balances detail vs file-size.
_JPEG_QUALITY = 85


class ScreenshotCapture:
    """Captures and persists violation screenshots.

    Parameters
    ----------
    capture_dir:
        Absolute or project-relative path where screenshots are saved.
        Defaults to ``<project>/reports/violation_captures``.
    cooldown:
        Minimum seconds between captures of the **same** violation type.
        Set to ``0`` to capture every violation unconditionally.
    """

    def __init__(
        self,
        capture_dir: str | Path | None = None,
        cooldown: float = 10.0,
        submission_id: str | None = None,
    ) -> None:
        self.capture_dir = Path(capture_dir) if capture_dir else _DEFAULT_CAPTURE_DIR
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown = cooldown
        self.submission_id = submission_id
        self._last_capture: dict[str, float] = {}

    # ── public API ──────────────────────────────────────────────

    def capture(
        self,
        frame: np.ndarray,
        violation_type: str,
        timestamp: str | None = None,
    ) -> str | None:
        """Save *frame* as a JPEG screenshot for *violation_type*.

        Parameters
        ----------
        frame:
            BGR image (numpy array) — typically the annotated frame.
        violation_type:
            e.g. ``"GAZE_AWAY"``, ``"MULTIPLE_FACES"``.
        timestamp:
            ISO-8601 or ``HH:MM:SS.mmm`` string used for the filename.
            Falls back to the current wall-clock time if not provided.

        Returns
        -------
        str | None
            The **relative path** from the ``reports/`` directory
            (e.g. ``violation_captures/GAZE_AWAY_20260703_140000_a1b2.jpg``).
            Returns ``None`` if the cooldown has not elapsed yet.
        """
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

        # Return path relative to reports/ so it is portable across
        # host ↔ container mount boundaries.
        relative = f"violation_captures/{filename}"
        return relative

    # ── internals ───────────────────────────────────────────────

    def _can_capture(self, violation_type: str) -> bool:
        """Return True if the cooldown for *violation_type* has elapsed."""
        if self.cooldown <= 0:
            return True
        last = self._last_capture.get(violation_type, 0.0)
        return (time.time() - last) >= self.cooldown

    @staticmethod
    def _safe_timestamp(ts: str | None) -> str:
        """Convert an arbitrary timestamp string into a filename-safe form."""
        if ts is None:
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            return ts
        # Strip characters that are illegal or awkward in filenames.
        return (
            ts.replace(":", "")
              .replace("-", "")
              .replace("T", "_")
              .replace("+", "_")
              .replace(".", "_")
              .replace(" ", "_")
        )
