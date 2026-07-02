import cv2
import os
from datetime import datetime

from models import AppConfig, ViolationType

class ViolationCapturer:
    def __init__(self, config: AppConfig):
        self.output_dir = os.path.join(config.global_.output_path, "violation_captures")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def capture_violation(self, frame, violation_type: ViolationType, timestamp: str | None = None):
        """Saves violation screenshot with metadata"""
        timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{violation_type.value}_{timestamp}.jpg"
        path = os.path.join(self.output_dir, filename)
        
        labeled_frame = frame.copy()
        cv2.putText(labeled_frame, f"{violation_type.value} - {timestamp}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imwrite(path, labeled_frame)
        return {
            'type': violation_type.value,
            'timestamp': timestamp,
            'image_path': os.path.abspath(path)
        }