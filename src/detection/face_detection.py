import cv2
import mediapipe as mp
from datetime import datetime
from typing import Optional

from models import AppConfig, AlertType, ViolationType
from .base import BaseDetector


class FaceDetector(BaseDetector):
    def __init__(self, config: AppConfig):
        self.mp_face_detection = mp.solutions.face_detection
        face_cfg = config.detection.face
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=face_cfg.min_confidence,
        )
        self.detection_interval = face_cfg.detection_interval
        self.min_confidence = face_cfg.min_confidence
        self.frame_count = 0
        self.face_present = False
        self.last_face_time = None
        self.face_disappeared_start = None

    def process(self, frame, **kwargs) -> bool:
        self.frame_count += 1
        if self.frame_count % self.detection_interval != 0:
            return self.face_present

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)

        current_time = datetime.now()
        detections = results.detections if results.detections else []
        face_found = any(d.score[0] > self.min_confidence for d in detections)

        if face_found:

            self.face_present = True
            self.last_face_time = current_time
            self.face_disappeared_start = None
            return True
        else:
            if self.face_present:
                self.face_disappeared_start = current_time

            self.face_present = False
            return False

    def close(self):
        """Close MediaPipe FaceDetection resources."""
        self.detector.close()