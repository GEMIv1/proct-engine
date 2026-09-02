from dataclasses import dataclass, field
from .enums import GazeDirection

@dataclass
class HeadPose:
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0


@dataclass
class PupilPosition:
    x: float = 0.5
    y: float = 0.5


@dataclass
class GazeState:
    gaze: GazeDirection = GazeDirection.UNCERTAIN
    gaze_conf: float = 0.0
    head_pose: HeadPose = field(default_factory=HeadPose)
    pupil_rel: PupilPosition = field(default_factory=PupilPosition)
