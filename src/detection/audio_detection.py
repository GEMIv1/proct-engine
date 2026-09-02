import pyaudio
import numpy as np
import threading
from collections import deque
from typing import Optional

from models import AppConfig, AlertType


class AudioMonitor:
    def __init__(self, config: AppConfig, alert_system=None):
        self.audio_config = config.detection.audio_monitoring
        self.sample_rate = self.audio_config.sample_rate
        self.chunk_size = 512  
        self.energy_threshold = self.audio_config.energy_threshold
        self.zcr_threshold = self.audio_config.zcr_threshold
        self.running = False
        self.audio_buffer: deque = deque(maxlen=15) 
        self.alert_system = alert_system



    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1)

    def _run(self):
        try:
            p = pyaudio.PyAudio()
        except Exception as e:
            return

        stream = None
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
        except Exception as e:
            p.terminate()
            return

        try:
            while self.running:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16)
                self.audio_buffer.append(audio)

                if self._is_voice(audio):
                    self._handle_voice_detection()

        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            p.terminate()

    def _is_voice(self, audio: np.ndarray) -> bool:
        audio_norm = audio / 32768.0

        energy = np.mean(audio_norm ** 2)
        if energy < self.energy_threshold:
            return False

        zcr = np.mean(np.abs(np.diff(np.sign(audio_norm))))
        if zcr > self.zcr_threshold:
            return False

        return True

    def _handle_voice_detection(self):
        if self.alert_system:
            self.alert_system.speak_alert(AlertType.VOICE_DETECTED)