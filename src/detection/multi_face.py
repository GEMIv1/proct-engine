import cv2
import mediapipe as mp
from typing import Optional

from models import AppConfig
from .base import BaseDetector


class MultiFaceDetector(BaseDetector):
    def __init__(self, config: AppConfig):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.7,
        )
        self.threshold = config.detection.multi_face.alert_threshold
        self.consecutive_frames = 0

    def process(self, frame, **kwargs) -> bool:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)

        detections = results.detections if results.detections else []
        high_conf_faces = sum(1 for d in detections if d.score[0] > 0.9)

        if high_conf_faces >= 2:
            self.consecutive_frames += 1
            if self.consecutive_frames >= self.threshold:
                return True
        else:
            self.consecutive_frames = 0

        return False

    def close(self):
        self.detector.close()