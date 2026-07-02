"""
ExamProctor — high-level facade over the proctoring detection pipelines (video & live).

Provides a unified interface for analyzing recorded video files and initiating
live webcam-based proctoring sessions, formatting the final outputs consistently.

Usage (Recorded Video)::

    from proctor import ExamProctor
    proctor = ExamProctor()
    result  = proctor.process_video("/path/to/video.mp4")

Usage (Live Camera Session)::

    from proctor import ExamProctor
    proctor = ExamProctor()
    result  = proctor.process_live(preview=True)
"""

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Union

from models import ViolationType
from utils.config_loader import load_config
from pipelines.video_pipeline import process_video
from pipelines.live_pipeline import process_live


class ExamProctor:
    """High-level facade around proctoring execution pipelines.

    Unifies the processing loop, violation accumulation, risk scoring,
    and report generation for both live webcam streams and pre-recorded videos.
    """

    def __init__(self):
        self.config = load_config()
        self.severity_map: dict[str, int] = self.config.global_.severity_levels

    def process_video(self, video_path: str, student_info: dict = None) -> dict:
        """Run the recorded video detection pipeline. (Retained for backward compatibility)."""
        return self.process(mode="video", source=video_path, student_info=student_info, preview=False)

    def process_live(self, student_info: dict = None, preview: bool = True) -> dict:
        """Run the live webcam proctoring session."""
        return self.process(mode="live", source=self.config.video.source, student_info=student_info, preview=preview)

    def process(self, mode: str, source: Union[str, int], student_info: dict = None, preview: bool = True) -> dict:
        """Core unified processor. Coordinates input modes and aggregates violation summaries."""
        if student_info is None:
            student_info = {
                "id": "WORKER_ANALYSIS" if mode == "video" else "STUDENT_001",
                "name": "Unknown Student",
                "exam": "Automated Analysis",
                "course": "N/A",
            }

        if mode == "video":
            result = process_video(
                video_path=str(source),
                config=self.config,
                preview=preview,
                save_output=False,
                student_info=student_info,
            )
        elif mode == "live":
            result = process_live(
                config=self.config,
                student_info=student_info,
                preview=preview,
            )
        else:
            raise ValueError(f"Unknown proctoring mode: {mode}")

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
