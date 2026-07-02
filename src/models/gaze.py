from dataclasses import dataclass, field
from .enums import GazeDirection

@dataclass
class HeadPose:
    """Rough head pose estimated from iris positions."""
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0


@dataclass
class PupilPosition:
    """Normalised pupil position in [0, 1] range."""
    x: float = 0.5
    y: float = 0.5


@dataclass
class GazeState:
    """Full gaze estimation result returned by GazeDetector.process()."""
    gaze: GazeDirection = GazeDirection.UNCERTAIN
    gaze_conf: float = 0.0
    head_pose: HeadPose = field(default_factory=HeadPose)
    pupil_rel: PupilPosition = field(default_factory=PupilPosition)
