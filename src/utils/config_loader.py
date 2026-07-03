"""
Centralised configuration loader.

Reads ``config/config.yaml`` and returns a fully‑typed :class:`AppConfig`
dataclass instance, replacing the duplicated ``load_config()`` helpers that
previously existed in both ``main.py`` and ``process_video.py``.
"""

from __future__ import annotations

from pathlib import Path
import os

import yaml

from models import (
    AlertConfig,
    AppConfig,
    AudioMonitoringConfig,
    DetectionConfig,
    EyeDetectionConfig,
    FaceDetectionConfig,
    GlobalConfig,
    MouthMonitoringConfig,
    MultiFaceConfig,
    ObjectDetectionConfig,
    VideoConfig,
    RabbitMQConfig,
)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load and parse ``config.yaml`` into an :class:`AppConfig` instance.

    Parameters
    ----------
    path:
        Optional override for the config file location.  Defaults to
        ``config/config.yaml`` relative to the working directory.

    Returns
    -------
    AppConfig
        A fully populated configuration dataclass.
    """
    config_path = Path(path) if path else _CONFIG_PATH

    with open(config_path, encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f)

    v = raw.get("video", {})
    video = VideoConfig(
        source=v.get("source", 0),
        resolution=v.get("resolution", [1280, 720]),
        fps=v.get("fps", 30),
        recording_path=v.get("recording_path", "./recordings"),
    )

    d = raw.get("detection", {})

    face_cfg = d.get("face", {})
    face = FaceDetectionConfig(
        detection_interval=face_cfg.get("detection_interval", 5),
        min_confidence=face_cfg.get("min_confidence", 0.8),
        face_absent_threshold=face_cfg.get("face_absent_threshold", 3),
    )

    eyes_cfg = d.get("eyes", {})
    eyes = EyeDetectionConfig(
        gaze_threshold=eyes_cfg.get("gaze_threshold", 2),
        iris_up_thresh=eyes_cfg.get("iris_up_thresh", 0.45),
        iris_down_thresh=eyes_cfg.get("iris_down_thresh", 0.35),
        gaze_left_thresh=eyes_cfg.get("gaze_left_thresh", 0.45),
        gaze_right_thresh=eyes_cfg.get("gaze_right_thresh", 0.60),
        max_frames_without_detection=eyes_cfg.get("max_frames_without_detection", 10),
        iris_pupil_blend=eyes_cfg.get("iris_pupil_blend", 0.7),
        head_pose_scale=eyes_cfg.get("head_pose_scale", 30.0),
        max_frame_dim=eyes_cfg.get("max_frame_dim", 640),
    )

    mouth_cfg = d.get("mouth_monitor", {})
    mouth = MouthMonitoringConfig(
        movement_threshold=mouth_cfg.get("movement_threshold", 3),
        open_threshold=mouth_cfg.get("open_threshold", 0.03),
        width_threshold=mouth_cfg.get("width_threshold", 0.2),
    )

    mf_cfg = d.get("multi_face", {})
    multi_face = MultiFaceConfig(
        alert_threshold=mf_cfg.get("alert_threshold", 5),
    )

    obj_cfg = d.get("objects", {})
    objects = ObjectDetectionConfig(
        model_path=obj_cfg.get("model_path", "models/yolo26s_finetuned.pt"),
        min_confidence=obj_cfg.get("min_confidence", 0.65),
        detection_interval=obj_cfg.get("detection_interval", 5),
        max_fps=obj_cfg.get("max_fps", 5),
    )

    audio_cfg = d.get("audio_monitoring", {})
    audio = AudioMonitoringConfig(
        enabled=audio_cfg.get("enabled", True),
        sample_rate=audio_cfg.get("sample_rate", 16000),
        energy_threshold=audio_cfg.get("energy_threshold", 0.001),
        zcr_threshold=audio_cfg.get("zcr_threshold", 0.35),
    )

    detection = DetectionConfig(
        face=face,
        eyes=eyes,
        mouth_monitor=mouth,
        multi_face=multi_face,
        objects=objects,
        audio_monitoring=audio,
    )

    al = raw.get("alert", {})
    alert = AlertConfig(
        cooldown=al.get("cooldown", 10),
    )

    g = raw.get("global", {})
    global_cfg = GlobalConfig(
        severity_levels=g.get("severity_levels", {}),
    )
    rmq = raw.get("rabbitmq", {})
    rabbitmq = RabbitMQConfig(
        host=os.environ.get("RABBITMQ_HOST", rmq.get("host", "localhost")),
        port=int(os.environ.get("RABBITMQ_PORT", rmq.get("port", 5672))),
        user=os.environ.get("RABBITMQ_USER", rmq.get("user", "guest")),
        password=os.environ.get("RABBITMQ_PASS", rmq.get("password", "guest")),
        vhost=os.environ.get("RABBITMQ_VHOST", rmq.get("vhost", "/")),
        request_queue=os.environ.get("RABBITMQ_REQUEST_QUEUE", rmq.get("request_queue", "cheating-detection-queue")),
        response_queue=os.environ.get("RABBITMQ_RESPONSE_QUEUE", rmq.get("response_queue", "cheating-detection-result-queue")),
        submissions_base_path=os.environ.get("SUBMISSIONS_BASE_PATH", rmq.get("submissions_base_path", "")),
    )

    return AppConfig(
        video=video,
        detection=detection,
        alert=alert,
        global_=global_cfg,
        rabbitmq=rabbitmq,
    )
