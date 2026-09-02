import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

from models import AppConfig, AlertType
from .base import BaseDetector


class _LandmarkIndex:
    UPPER_INNER_LIP = 13
    LOWER_INNER_LIP = 14
    RIGHT_CORNER = 78
    LEFT_CORNER = 306


class MouthMonitor(BaseDetector):
    def __init__(self, config: AppConfig):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.mouth_threshold = config.detection.mouth_monitor.movement_threshold
        self.open_threshold = config.detection.mouth_monitor.open_threshold
        self.width_threshold = config.detection.mouth_monitor.width_threshold
        self.mouth_movement_count = 0

    def close(self):
        self.face_mesh.close()

    def process(self, frame: np.ndarray, **kwargs) -> bool:
        landmarks = self._get_face_landmarks(frame)
        if landmarks is None:
            return False

        moving = self._is_mouth_moving(landmarks)
        self._update_alert_state(moving)
        return moving

    def _get_face_landmarks(self, frame: np.ndarray):
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]

    def _is_mouth_moving(self, landmarks) -> bool:
        mouth_open = (
            landmarks.landmark[_LandmarkIndex.LOWER_INNER_LIP].y
            - landmarks.landmark[_LandmarkIndex.UPPER_INNER_LIP].y
        )
        mouth_width = abs(
            landmarks.landmark[_LandmarkIndex.LEFT_CORNER].x
            - landmarks.landmark[_LandmarkIndex.RIGHT_CORNER].x
        )
        return mouth_open > self.open_threshold or mouth_width > self.width_threshold

    def _update_alert_state(self, moving: bool) -> None:
        if moving:
            self.mouth_movement_count += 1
            if self.mouth_movement_count > self.mouth_threshold:
                self.mouth_movement_count = 0
        else:
            self.mouth_movement_count = max(0, self.mouth_movement_count - 1)