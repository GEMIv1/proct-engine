"""detection — per-frame CV/audio analysis detectors."""

from .base import BaseDetector
from .face_detection import FaceDetector
from .gaze_detector import GazeDetector
from .mouth_monitor import MouthMonitor
from .multi_face import MultiFaceDetector
from .object_detection import ObjectDetector
from .audio_detection import AudioMonitor

__all__ = [
    "BaseDetector",
    "FaceDetector",
    "GazeDetector",
    "MouthMonitor",
    "MultiFaceDetector",
    "ObjectDetector",
    "AudioMonitor",
]
