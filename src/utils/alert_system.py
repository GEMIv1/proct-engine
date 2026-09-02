import os
import tempfile
import pygame
import threading
import time
from typing import Union
from gtts import gTTS

from models import AppConfig, AlertType, ViolationType


class AlertSystem:
    def __init__(self, config: AppConfig):
        pygame.mixer.init()
        self.alert_cooldown = config.alert.cooldown
        self.last_alert_time: dict = {}

        self.alerts: dict[Union[ViolationType, AlertType], str] = {
            ViolationType.FACE_DISAPPEARED: "Please look at the screen",
            ViolationType.MULTIPLE_FACES: "We detected multiple people",
            ViolationType.OBJECT_DETECTED: "Unauthorized object detected",
            ViolationType.GAZE_AWAY: "Please focus on your screen",
            ViolationType.MOUTH_MOVING: "Please maintain silence during exam",
            AlertType.SPEECH_VIOLATION: "Speaking during exam is not allowed",
            AlertType.VOICE_DETECTED: "We detected voice, Please maintain silence during the exam",
        }

    def _can_alert(self, alert_type: Union[ViolationType, AlertType]) -> bool:
        current_time = time.time()
        last_time = self.last_alert_time.get(alert_type, 0)
        return (current_time - last_time) >= self.alert_cooldown

    def speak_alert(self, alert_type: Union[ViolationType, AlertType]):
        if not self._can_alert(alert_type):
            return

        self.last_alert_time[alert_type] = time.time()

        def _play_audio():
            try:
                if alert_type in self.alerts:
                    tts = gTTS(text=self.alerts[alert_type], lang='en')

                    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
                        temp_path = fp.name
                        tts.save(temp_path)

                    pygame.mixer.music.load(temp_path)
                    pygame.mixer.music.play()

                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)

                    os.unlink(temp_path)
            except Exception as e:
                print(f"Audio alert failed: {str(e)}")
        threading.Thread(target=_play_audio, daemon=True).start()