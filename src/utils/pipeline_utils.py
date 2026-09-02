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


def build_detectors(config: "AppConfig") -> tuple[BaseDetector, BaseDetector, BaseDetector, BaseDetector, BaseDetector]:
    face = FaceDetector(config)
    gaze = GazeDetector(config, smoothing=5)
    mouth = MouthMonitor(config)
    multi = MultiFaceDetector(config)
    obj = ObjectDetector(config)
    return face, gaze, mouth, multi, obj
