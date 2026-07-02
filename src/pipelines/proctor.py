"""
ExamProctor — high-level facade over the video detection pipeline.

Used by the RabbitMQ worker to process a video and return a result dict
that maps to :class:`~models.messaging.CheatingDetectionResult` fields.

Usage::

    from proctor import ExamProctor
    proctor = ExamProctor()
    result  = proctor.process_video("/path/to/video.mp4")
"""

import json
from collections import Counter
from datetime import datetime, timezone

from models import ViolationType
from utils.config_loader import load_config
from video_pipeline import process_video


class ExamProctor:
    """High-level facade around :func:`~video_pipeline.process_video`.

    Instantiated by the RabbitMQ worker to process a single video and
    return a result dict that maps directly to
    :class:`~models.messaging.CheatingDetectionResult` fields.
    """

    def __init__(self):
        self.config = load_config()
        # Severity levels come exclusively from config — no fallback duplication
        self.severity_map: dict[str, int] = self.config.reporting.severity_levels

    def process_video(self, video_path: str, student_info: dict = None) -> dict:
        """Run the full detection pipeline and return a result dict."""

        if student_info is None:
            student_info = {
                "id": "WORKER_ANALYSIS",
                "name": "Unknown Student",
                "exam": "Automated Analysis",
                "course": "N/A",
            }

        result = process_video(
            video_path=video_path,
            config=self.config,
            preview=False,
            save_output=False,
            student_info=student_info,
        )

        violations = result["violations"]

        # ── Risk score (0–100) ──────────────────────────────────
        total_severity = sum(
            self.severity_map.get(v.type.value, 1) for v in violations
        )
        risk_score = min(100.0, round(total_severity * 2.0, 2))

        # ── Suspicious-behaviour summary ────────────────────────
        violation_counts = Counter(v.type.value for v in violations)
        suspicious = [
            {"type": vtype, "count": count}
            for vtype, count in violation_counts.most_common()
        ]

        # ── Detected objects ────────────────────────────────────
        object_violations = [
            v for v in violations if v.type == ViolationType.OBJECT_DETECTED
        ]
        detected_objects = [v.metadata for v in object_violations]

        # ── Full report JSON ────────────────────────────────────
        report_data = {
            "violations": [v.to_dict() for v in violations],
            "summary": {
                "total_violations": len(violations),
                "frames_processed": result.get("frames_processed", 0),
                "violation_counts": dict(violation_counts),
                "risk_score": risk_score,
            },
        }

        return {
            "risk_score": risk_score,
            "suspicious_behavior": json.dumps(suspicious),
            "detected_objects": json.dumps(detected_objects),
            "report_json": json.dumps(report_data),
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "video_quality": None,
            "calibration_quality": None,
        }
