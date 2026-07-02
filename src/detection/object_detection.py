import cv2
import torch
from ultralytics import YOLO
from datetime import datetime
from typing import Optional

from models import AppConfig, AlertType
from .base import BaseDetector


class ObjectDetector(BaseDetector):
    def __init__(self, config: AppConfig):
        self.obj_config = config.detection.objects
        self.model = None
        self.class_map = {
            0: 'book',
            1: 'cell phone',
            2: 'headphone',
        }
        self.detection_interval = self.obj_config.detection_interval
        self.frame_count = 0
        self._initialize_model()
        self.last_detection_time = datetime.now()

    def _initialize_model(self):
        """Initialize optimized YOLO model using path from config."""
        try:
            self.model = YOLO(self.obj_config.model_path)

            self.model.overrides['conf'] = self.obj_config.min_confidence
            self.model.overrides['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.overrides['iou'] = 0.45   # Slightly higher IOU threshold

            dummy_input = torch.zeros((1, 3, 640, 480)).to(self.model.device)
            self.model(dummy_input)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize object detector: {str(e)}")

    def process(self, frame, visualize=False, **kwargs) -> bool:
        """Optimized object detection with frame skipping."""
        current_time = datetime.now()
        time_since_last = (current_time - self.last_detection_time).total_seconds()

        if time_since_last < (1.0 / self.obj_config.max_fps):
            return False

        try:
            orig_h, orig_w = frame.shape[:2]
            new_w = 320
            new_h = int(orig_h * (new_w / orig_w))
            resized_frame = cv2.resize(frame, (new_w, new_h))

            results = self.model(resized_frame, verbose=False)

            detected = False
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)

                    if cls in self.class_map and conf > self.obj_config.min_confidence:
                        detected = True
                        label = self.class_map[cls]

                        if visualize:
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1 = int(x1 * (orig_w / new_w))
                            y1 = int(y1 * (orig_h / new_h))
                            x2 = int(x2 * (orig_w / new_w))
                            y2 = int(y2 * (orig_h / new_h))

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            self.last_detection_time = current_time
            return detected

        except Exception as e:
            return False

    def close(self):
        """Clean up YOLO detector resources."""
        self.model = None