from dataclasses import dataclass
from .enums import GazeDirection

@dataclass
class DetectionResult:
    face_present: bool = False
    gaze_direction: GazeDirection = GazeDirection.ON_SCREEN
    gaze_conf: float = 0.0
    mouth_moving: bool = False
    multiple_faces: bool = False
    objects_detected: bool = False
    timestamp: str = ""
