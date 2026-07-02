from dataclasses import dataclass
from typing import Optional


@dataclass
class CheatingDetectionRequest:
    """Incoming message from the .NET backend requesting cheating analysis.

    Field names use snake_case; the worker converts PascalCase JSON keys
    from .NET before constructing this dataclass.
    """

    quiz_submission_id: str = ""
    quiz_id: str = ""
    student_user_id: str = ""
    user_full_name: str = ""
    student_id_in_course: Optional[str] = None
    video_path: str = ""
    video_size: int = 0
    video_duration: str = ""
    submitted_at: str = ""


@dataclass
class CheatingDetectionResult:
    """Result sent back to the .NET backend after processing.

    Field names use snake_case; the worker converts to PascalCase
    before publishing the JSON response.
    """

    quiz_submission_id: str = ""
    risk_score: float = 0.0
    suspicious_behavior: str = ""
    detected_objects: str = ""
    report_json: str = ""
    processed_at: str = ""
    video_quality: Optional[str] = None
    calibration_quality: Optional[str] = None
