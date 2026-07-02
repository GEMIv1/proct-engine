import cv2
import os
from datetime import datetime

from models import AppConfig, RecordingResult

class VideoRecorder:
    def __init__(self, config: AppConfig):
        self.recording_path = config.video.recording_path
        self.resolution = tuple(config.video.resolution)
        self.fps = config.video.fps
        self.writer = None
        self.filename = None
        self.frame_count = 0
        self.start_time = datetime.now()
        
    def start_recording(self):
        if not os.path.exists(self.recording_path):
            os.makedirs(self.recording_path)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(self.recording_path, f"webcam_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.filename,
            fourcc,
            self.fps,
            self.resolution
        )
        
    def record_frame(self, frame):
        if self.writer:
            self.writer.write(frame)
            self.frame_count += 1
            
    def stop_recording(self) -> RecordingResult | None:
        if self.writer:
            self.writer.release()
            self.writer = None
            duration = (datetime.now() - self.start_time).total_seconds()
            return RecordingResult(
                filename=self.filename,
                frame_count=self.frame_count,
                duration=duration,
                fps=self.frame_count / duration if duration > 0 else 0,
            )
        return None