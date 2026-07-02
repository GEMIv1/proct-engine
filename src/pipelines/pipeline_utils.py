"""
pipeline_utils — shared helpers used by both pipeline entry points.

Provides :func:`build_detectors` to construct and wire all detectors in
one place, eliminating the copy-paste that previously existed in both
``live_pipeline.py`` and ``video_pipeline.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from detection.base import BaseDetector
from detection.face_detection import FaceDetector
from detection.gaze_detector import GazeDetector
from detection.mouth_monitor import MouthMonitor
from detection.multi_face import MultiFaceDetector
from detection.object_detection import ObjectDetector

if TYPE_CHECKING:
    from models import AppConfig
    from utils.alert_logger import AlertLogger


def build_detectors(
    config: "AppConfig",
    alert_logger: "AlertLogger",
) -> tuple[BaseDetector, BaseDetector, BaseDetector, BaseDetector, BaseDetector]:
    """Construct and wire all CV detectors.

    Parameters
    ----------
    config:
        Fully-loaded :class:`~models.AppConfig` instance.
    alert_logger:
        :class:`~utils.alert_logger.AlertLogger` instance that will receive
        internal detector alert events.

    Returns
    -------
    tuple
        ``(face, gaze, mouth, multi_face, object_detector)``
    """
    face = FaceDetector(config, alert_logger=alert_logger)
    gaze = GazeDetector(config, smoothing=5)
    mouth = MouthMonitor(config, alert_logger=alert_logger)
    multi = MultiFaceDetector(config, alert_logger=alert_logger)
    obj = ObjectDetector(config, alert_logger=alert_logger)
    return face, gaze, mouth, multi, obj
