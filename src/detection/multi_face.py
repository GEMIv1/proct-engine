import cv2
import mediapipe as mp
from typing import Optional

from models import AppConfig, AlertType, ViolationType
from utils.alert_logger import AlertLogger
from .base import BaseDetector


class MultiFaceDetector(BaseDetector):
    def __init__(self, config: AppConfig, alert_logger: Optional[AlertLogger] = None):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,       # 0 = short-range (< 2m), 1 = full-range (< 5m)
            min_detection_confidence=0.7,
        )
        self.threshold = config.detection.multi_face.alert_threshold
        self.consecutive_frames = 0
        self.alert_logger = alert_logger

    def process(self, frame, **kwargs) -> bool:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)

        detections = results.detections if results.detections else []
        # Count faces with high confidence
        high_conf_faces = sum(1 for d in detections if d.score[0] > 0.9)

        if high_conf_faces >= 2:
            self.consecutive_frames += 1
            if self.consecutive_frames >= self.threshold and self.alert_logger:
                self.alert_logger.log_alert(
                    ViolationType.MULTIPLE_FACES,
                    f"Detected {high_conf_faces} faces for {self.consecutive_frames} frames"
                )
                return True
        else:
            self.consecutive_frames = 0

        return False

    def close(self):
        """Close MediaPipe FaceDetection resources."""
        self.detector.close()