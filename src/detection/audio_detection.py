import pyaudio
import numpy as np
import threading
from collections import deque
from typing import Optional

from models import AppConfig, AlertType


class AudioMonitor:
    def __init__(
        self,
        config: AppConfig,
        alert_system=None,
    ):
        """
        Parameters
        ----------
        config:
            Application configuration.
        alert_system:
            Optional :class:`~utils.alert_system.AlertSystem` for voice alerts.
            Accepts ``None`` for headless / test environments.
        """
        self.audio_config = config.detection.audio_monitoring
        self.sample_rate = self.audio_config.sample_rate
        self.chunk_size = 512  # 32ms chunks for low latency
        self.energy_threshold = self.audio_config.energy_threshold
        self.zcr_threshold = self.audio_config.zcr_threshold
        self.running = False
        self.audio_buffer: deque = deque(maxlen=15)  # 480ms buffer
        self.alert_system = alert_system

        if self.audio_config.whisper_enabled:
            import whisper  # optional heavy dependency
            self.whisper_model = whisper.load_model(self.audio_config.whisper_model)
        else:
            self.whisper_model = None

    def start(self):
        """Start audio monitoring thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop audio monitoring."""
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1)

    def _run(self):
        """Main audio processing loop."""
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
        """Ultra-fast voice detection."""
        audio_norm = audio / 32768.0

        # 1. Energy detection
        energy = np.mean(audio_norm ** 2)
        if energy < self.energy_threshold:
            return False

        # 2. Zero-crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(audio_norm))))
        if zcr > self.zcr_threshold:
            return False

        return True

    def _handle_voice_detection(self):
        """Process detected voice."""
        if self.alert_system:
            self.alert_system.speak_alert(AlertType.VOICE_DETECTED)

        if self.audio_config.whisper_enabled and self.whisper_model:
            self._process_with_whisper()

    def _process_with_whisper(self):
        """Optional Whisper processing."""
        try:
            audio = np.concatenate(self.audio_buffer)
            result = self.whisper_model.transcribe(
                audio.astype(np.float32) / 32768.0,
                fp16=False,
                language='en'
            )

            if self.alert_system:
                    self.alert_system.speak_alert(AlertType.SPEECH_VIOLATION)

        except Exception as e:
            pass