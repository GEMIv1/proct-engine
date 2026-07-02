from dataclasses import dataclass, field

@dataclass
class VideoConfig:
    source: int = 0
    resolution: list[int] = field(default_factory=lambda: [1280, 720])
    fps: int = 30
    recording_path: str = "./recordings"


@dataclass
class FaceDetectionConfig:
    detection_interval: int = 5
    min_confidence: float = 0.8
    face_absent_threshold: int = 3  # seconds before flagging face disappeared


@dataclass
class EyeDetectionConfig:
    gaze_threshold: int = 2
    iris_up_thresh: float = 0.45
    iris_down_thresh: float = 0.35
    gaze_left_thresh: float = 0.45
    gaze_right_thresh: float = 0.60
    max_frames_without_detection: int = 10
    iris_pupil_blend: float = 0.7
    head_pose_scale: float = 30.0
    max_frame_dim: int = 640


@dataclass
class MouthMonitoringConfig:
    movement_threshold: int = 3
    open_threshold: float = 0.03
    width_threshold: float = 0.2


@dataclass
class MultiFaceConfig:
    alert_threshold: int = 5


@dataclass
class ObjectDetectionConfig:
    model_path: str = "models/yolo26s_finetuned.pt"
    min_confidence: float = 0.65
    detection_interval: int = 5
    max_fps: int = 5


@dataclass
class AudioMonitoringConfig:
    enabled: bool = True
    sample_rate: int = 16000
    energy_threshold: float = 0.001
    zcr_threshold: float = 0.35
    whisper_enabled: bool = False
    whisper_model: str = "tiny.en"


@dataclass
class AlertConfig:
    cooldown: int = 10



@dataclass
class DetectionConfig:
    face: FaceDetectionConfig = field(default_factory=FaceDetectionConfig)
    eyes: EyeDetectionConfig = field(default_factory=EyeDetectionConfig)
    mouth_monitor: MouthMonitoringConfig = field(default_factory=MouthMonitoringConfig)
    multi_face: MultiFaceConfig = field(default_factory=MultiFaceConfig)
    objects: ObjectDetectionConfig = field(default_factory=ObjectDetectionConfig)
    audio_monitoring: AudioMonitoringConfig = field(default_factory=AudioMonitoringConfig)


@dataclass
class GlobalConfig:
    severity_levels: dict[str, int] = field(default_factory=lambda: {
        "FACE_DISAPPEARED": 1,
        "GAZE_AWAY": 2,
        "MOUTH_MOVING": 3,
        "MULTIPLE_FACES": 4,
        "OBJECT_DETECTED": 5,
        "AUDIO_DETECTED": 3,
    })


@dataclass
class RabbitMQConfig:
    host: str = "localhost"
    port: int = 5672
    user: str = "guest"
    password: str = "guest"
    vhost: str = "/"
    request_queue: str = "cheating-detection-queue"
    response_queue: str = "cheating-detection-result-queue"
    submissions_base_path: str = ""


@dataclass
class AppConfig:
    """Top‑level application configuration – aggregates every section."""
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    global_: GlobalConfig = field(default_factory=GlobalConfig)
    rabbitmq: RabbitMQConfig = field(default_factory=RabbitMQConfig)
