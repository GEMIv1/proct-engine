from __future__ import annotations

import cv2
import numpy as np

from models import DetectionResult, GazeState


def annotate_frame(frame: np.ndarray, results: DetectionResult, gaze_state: GazeState) -> np.ndarray:
    out = frame.copy()
    y_offset = 30
    line_height = 30

    gaze = gaze_state.gaze.value
    gaze_conf = gaze_state.gaze_conf
    yaw = gaze_state.head_pose.yaw
    pitch = gaze_state.head_pose.pitch

    cv2.putText(
        out,
        f"gaze:{gaze} ({gaze_conf:.0%}) yaw:{yaw:.1f} pitch:{pitch:.1f}",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    y_offset += line_height

    status_items = [
        f"Face: {'Present' if results.face_present else 'Absent'}",
        f"Mouth: {'Moving' if results.mouth_moving else 'Still'}",
    ]
    for item in status_items:
        cv2.putText(
            out, item, (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        y_offset += line_height

    if results.multiple_faces:
        cv2.putText(
            out, "Multiple Faces Detected!", (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )
        y_offset += line_height

    if results.objects_detected:
        cv2.putText(
            out, "Suspicious Object Detected!", (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )
        y_offset += line_height

    cv2.putText(
        out, results.timestamp,
        (out.shape[1] - 250, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )

    return out
