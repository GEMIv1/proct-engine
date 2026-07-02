from enum import Enum


class ViolationType(str, Enum):
    """Logged proctoring violations — these appear in violation reports."""
    FACE_DISAPPEARED = "FACE_DISAPPEARED"
    GAZE_AWAY = "GAZE_AWAY"
    MOUTH_MOVING = "MOUTH_MOVING"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    OBJECT_DETECTED = "OBJECT_DETECTED"
    AUDIO_DETECTED = "AUDIO_DETECTED"


class AlertType(str, Enum):
    """Internal alert / event types — used for voice alerts and log entries
    but do NOT appear as proctoring violations in reports."""
    VOICE_DETECTED = "VOICE_DETECTED"
    SPEECH_VIOLATION = "SPEECH_VIOLATION"
    FORBIDDEN_OBJECT = "FORBIDDEN_OBJECT"
    OBJECT_DETECTION_ERROR = "OBJECT_DETECTION_ERROR"
    WHISPER_ERROR = "WHISPER_ERROR"
    MOUTH_MOVEMENT = "MOUTH_MOVEMENT"  # internal detector signal


class GazeDirection(str, Enum):
    """Possible gaze directions from the gaze detector."""
    ON_SCREEN = "on_screen"
    OFF_LEFT = "off_left"
    OFF_RIGHT = "off_right"
    UP = "up"
    DOWN = "down"
    UNCERTAIN = "uncertain"

    @property
    def is_away(self) -> bool:
        """Return True if this direction counts as looking away."""
        return self in (
            GazeDirection.OFF_LEFT,
            GazeDirection.OFF_RIGHT,
            GazeDirection.UP,
            GazeDirection.DOWN,
        )
