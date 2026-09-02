<div align="center">

# ProctEngine

**AI-Powered Online Exam Proctoring System**

A real-time, multi-modal cheating detection engine built with computer vision, deep learning, and audio analysis. ProctEngine monitors exam sessions through video feeds — either live webcam streams or pre-recorded submissions — and produces structured violation reports with risk scores.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![YOLO v26](https://img.shields.io/badge/YOLO-v26-00FFFF?logo=yolo&logoColor=white)](https://docs.ultralytics.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-4285F4?logo=google&logoColor=white)](https://mediapipe.dev)
[![RabbitMQ](https://img.shields.io/badge/RabbitMQ-Worker-FF6600?logo=rabbitmq&logoColor=white)](https://www.rabbitmq.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)

</div>

---

## Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Detection Modules](#-detection-modules)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#%EF%B8%8F-configuration)
- [Usage](#-usage)
  - [Live Proctoring](#1-live-proctoring-webcam)
  - [Video Analysis](#2-recorded-video-analysis)
  - [RabbitMQ Worker](#3-rabbitmq-worker-production)
- [Docker Deployment](#-docker-deployment)
- [Model Finetuning](#-model-finetuning)
- [API Reference](#-api-reference)
- [Risk Scoring](#-risk-scoring)

---

## Overview

ProctEngine is a modular exam proctoring backend that performs frame-by-frame analysis on exam session video to detect a wide range of cheating behaviors. It operates in three modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Live** | Real-time webcam monitoring with voice alerts | In-person or remote live exams |
| **Video** | Offline analysis of pre-recorded submissions | Asynchronous post-exam review |
| **Worker** | Message-driven processing via RabbitMQ | Scalable production deployment with a .NET backend |

The system produces a structured JSON report containing timestamped violations, captured screenshot evidence, an aggregate risk score (0–100), and a per-violation-type summary.

---

## Key Features

- **Face Presence Detection** — Flags when the examinee's face disappears from frame for a configurable duration.
- **Gaze Tracking** — Iris-level gaze estimation via MediaPipe FaceMesh with calibration, smoothing, and directional classification (left, right, up, down).
- **Mouth Movement Monitoring** — Detects sustained mouth movement indicating potential verbal communication.
- **Multi-Face Detection** — Identifies when additional persons appear in the camera view.
- **Forbidden Object Detection** — Fine-tuned YOLOv26 model detects books, cell phones, and headphones in the frame.
- **Audio Monitoring** — Real-time voice/speech detection using energy and zero-crossing-rate analysis (live mode only).
- **Voice Alerts** — Text-to-speech warnings played back to the student during live sessions.
- **Violation Screenshots** — Automatic frame capture with cooldown-based deduplication, stored as JPEG evidence.
- **Risk Scoring** — Severity-weighted aggregate score from 0 to 100.
- **RabbitMQ Integration** — Message-driven worker for seamless integration with a .NET LMS backend.
- **Docker Support** — Single-command container deployment with CPU-based PyTorch.

---

## Architecture

![High-Level Architecture](docs/High%20level%20Architecture.png)

> The [`docs/`](docs/) directory also contains a **Sequence Diagram** illustrating the full detection flow and a detailed **Class Diagram** covering all modules and their relationships.

---

## Detection Modules

### Face Detector
Detects primary face presence using MediaPipe. If the face is absent for longer than `face_absent_threshold` seconds, a `FACE_DISAPPEARED` violation is raised.

### Gaze Detector
Uses MediaPipe FaceMesh with refined iris landmarks (468 mesh points) to compute gaze direction. Features include:
- **Iris-relative normalization** within the eye socket for position-invariant tracking
- **Pupil-iris blending** (`iris_pupil_blend`) for robustness against lighting changes
- **Moving-average smoothing** to reduce jitter
- **Optional calibration** to learn the user's baseline gaze center
- **Head-pose estimation** (pitch/yaw) derived from eye positions

Classified directions: `ON_SCREEN`, `OFF_LEFT`, `OFF_RIGHT`, `UP`, `DOWN`, `UNCERTAIN`.

### Mouth Monitor
Tracks mouth landmark distances (open ratio + width ratio) over consecutive frames to detect sustained mouth movement indicative of talking.

### Multi-Face Detector
Detects the presence of more than one face in the frame, flagging `MULTIPLE_FACES` violations for potential outside assistance.

### Object Detector
A **fine-tuned YOLOv26s** model trained to detect forbidden exam objects:

| Class ID | Object |
|----------|--------|
| 0 | Book |
| 1 | Cell Phone |
| 2 | Headphone |

The model is fine-tuned on a custom Roboflow dataset with 6 classes (book, cell phone, headphone, laptop, person, tv) and runs with frame-rate limiting and downscaled inference (320px width) for performance.

### Audio Monitor
Runs in a background thread (live mode only) capturing 512-sample chunks at 16 kHz. Voice detection uses a two-stage filter:
1. **Energy threshold** — rejects silence
2. **Zero-crossing rate** — rejects non-speech noise

Triggers a text-to-speech alert when voice is detected.

---

## Project Structure

```
ProctEngine/
├── config/
│   └── config.yaml              # Central configuration file
├── docs/
│   ├── High level Architecture.png
│   ├── Class Diagram.png
│   └── Sequence Diagram.png
├── finetune/
│   ├── Finetune_Online_proctoring_YOLO26.ipynb   # Training notebook
│   └── data/                    # Roboflow dataset (train/valid/test)
├── models/                      # YOLO weight files (.pt) — git-ignored
├── recordings/                  # Saved webcam recordings — git-ignored
├── reports/                     # Generated violation reports & screenshots
│   └── violation_captures/      # JPEG evidence frames
├── src/
│   ├── detection/               # Per-frame detection modules
│   │   ├── base.py              # BaseDetector abstract class
│   │   ├── face_detection.py    # Face presence detector
│   │   ├── gaze_detector.py     # Iris-level gaze tracking
│   │   ├── mouth_monitor.py     # Mouth movement detector
│   │   ├── multi_face.py        # Multiple face detector
│   │   ├── object_detection.py  # YOLO-based object detection
│   │   └── audio_detection.py   # Audio/voice monitoring
│   ├── messaging/
│   │   └── rabbitmq_worker.py   # RabbitMQ consumer/publisher worker
│   ├── models/                  # Data models & enums
│   │   ├── enums.py             # ViolationType, AlertType, GazeDirection
│   │   ├── detection.py         # DetectionResult dataclass
│   │   ├── gaze.py              # GazeState, HeadPose, PupilPosition
│   │   ├── session.py           # Session-related models
│   │   ├── settings.py          # AppConfig & sub-config dataclasses
│   │   └── messaging.py         # CheatingDetectionRequest/Result
│   ├── pipelines/
│   │   ├── proctor.py           # ExamProctor facade (unified API)
│   │   ├── live_pipeline.py     # Live webcam pipeline
│   │   └── video_pipeline.py    # Recorded video pipeline
│   └── utils/
│       ├── alert_system.py      # TTS voice alert system (gTTS + pygame)
│       ├── annotation.py        # Frame overlay/annotation rendering
│       ├── config_loader.py     # YAML → AppConfig parser
│       ├── pipeline_utils.py    # Shared detector builder
│       ├── screenshot_capture.py# Violation frame capture with cooldown
│       └── video_utils.py       # Video recording utilities
├── test_data/                   # Test video files — git-ignored
├── docker-compose.yml
├── dockerfile
├── requirements.txt
└── .gitignore
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| pip | Latest |
| RabbitMQ | 3.x+ (for worker mode) |
| Docker & Docker Compose | Latest (for containerized deployment) |
| Webcam | Any USB/built-in camera (for live mode) |
| Microphone | Any input device (for audio monitoring in live mode) |

### Platform Notes
- **Windows**: Requires `pywin32>=300` (auto-installed via `requirements.txt`).
- **Linux/macOS**: Requires system libraries for OpenCV (`libgl1`, `libglib2.0-0`) and PortAudio (`libportaudio2`).

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/ProctEngine.git
cd ProctEngine
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install PyTorch

Install PyTorch separately based on your hardware:

```bash
# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.x (NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download Model Weights

Place YOLO model weights in the `models/` directory. The system uses `yolo26s_finetuned.pt` by default (configured in `config/config.yaml`).

> **Note:** Model weights (`.pt` files) are git-ignored due to their size. Use Git LFS, DVC, or cloud storage for version control.

---

## Configuration

All settings are centralized in [`config/config.yaml`](config/config.yaml). The configuration is loaded into a typed `AppConfig` dataclass hierarchy via `utils/config_loader.py`.

### Video Settings

```yaml
video:
  source: 0                   # 0 = default webcam, or path to video file
  resolution: [1280, 720]
  fps: 30
  recording_path: "./recordings"
```

### Detection Thresholds

```yaml
detection:
  face:
    detection_interval: 5     # frames between face checks
    min_confidence: 0.8
    face_absent_threshold: 3  # seconds before FACE_DISAPPEARED violation
  eyes:
    gaze_threshold: 2         # seconds of sustained gaze-away before violation
    iris_up_thresh: 0.45
    iris_down_thresh: 0.35
    gaze_left_thresh: 0.45
    gaze_right_thresh: 0.60
    iris_pupil_blend: 0.7     # weight: iris-relative vs absolute pupil
    head_pose_scale: 30.0
    max_frame_dim: 640        # downscale for MediaPipe performance
  mouth_monitor:
    movement_threshold: 3     # consecutive frames of movement
    open_threshold: 0.03
    width_threshold: 0.2
  multi_face:
    alert_threshold: 5        # frames before alerting
  objects:
    model_path: "models/yolo26s_finetuned.pt"
    min_confidence: 0.65
    detection_interval: 5
    max_fps: 5                # rate-limit inference
  audio_monitoring:
    enabled: true
    sample_rate: 16000
    energy_threshold: 0.001
    zcr_threshold: 0.35
```

### Alert & Severity

```yaml
alert:
  cooldown: 10                # seconds between same alert/screenshot

global:
  severity_levels:            # weights for risk score calculation
    FACE_DISAPPEARED: 1
    GAZE_AWAY: 2
    MOUTH_MOVING: 3
    MULTIPLE_FACES: 4
    OBJECT_DETECTED: 5
    AUDIO_DETECTED: 3
```

### RabbitMQ (Worker Mode)

```yaml
rabbitmq:
  host: "localhost"
  port: 5672
  user: "admin"
  password: "admin"
  vhost: "/"
  request_queue: "cheating-detection-queue"
  response_queue: "cheating-detection-result-queue"
  submissions_base_path: ""
```

RabbitMQ settings can be overridden with environment variables: `RABBITMQ_HOST`, `RABBITMQ_PORT`, `RABBITMQ_USER`, `RABBITMQ_PASS`, `RABBITMQ_VHOST`, `SUBMISSIONS_BASE_PATH`.

---

## Usage

### 1. Live Proctoring (Webcam)

Start a real-time proctoring session with webcam preview and audio monitoring:

```python
from pipelines.proctor import ExamProctor

proctor = ExamProctor()
result = proctor.process_live(
    student_info={
        "id": "STU_001",
        "name": "Jane Doe",
        "exam": "Midterm Exam",
        "course": "CS101",
    },
    preview=True,   # show annotated video window
)

print(f"Risk Score: {result['risk_score']}")
print(f"Violations: {result['suspicious_behavior']}")
```

Or run the pipeline directly:

```bash
cd src
python -m pipelines.live_pipeline
```

Press **Q** to stop the live session.

### 2. Recorded Video Analysis

Analyze a pre-recorded exam video via CLI:

```bash
cd src
python -m pipelines.video_pipeline path/to/exam_video.mp4 --preview
```

**CLI Options:**

| Flag | Description |
|------|-------------|
| `--preview` | Display annotated video playback during processing |
| `--save-output` | Save the annotated video to `recordings/` |
| `--student-id ID` | Associate a student ID with the report |
| `--student-name NAME` | Associate a student name with the report |

**Programmatic usage:**

```python
from pipelines.proctor import ExamProctor

proctor = ExamProctor()
result = proctor.process_video("test_data/test1_cheating.mp4")
```

### 3. RabbitMQ Worker (Production)

The worker consumes messages from a RabbitMQ queue, processes videos, and publishes results back. This is designed for integration with a .NET LMS backend.

```bash
cd src
python -m messaging.rabbitmq_worker
```

**Request Message Format** (JSON on `cheating-detection-queue`):

```json
{
  "QuizSubmissionId": "sub_12345",
  "QuizId": "quiz_001",
  "StudentUserId": "user_42",
  "UserFullName": "Jane Doe",
  "VideoPath": "submissions/exam_video.mp4",
  "VideoSize": 10485760,
  "VideoDuration": "00:15:00",
  "SubmittedAt": "2026-09-01T14:30:00Z"
}
```

**Response Message Format** (JSON on `cheating-detection-result-queue`):

```json
{
  "QuizSubmissionId": "sub_12345",
  "RiskScore": 42.0,
  "SuspiciousBehavior": "[{\"type\": \"GAZE_AWAY\", \"count\": 5}]",
  "DetectedObjects": "[{\"video_time\": 120.5}]",
  "ReportJson": "{...}",
  "ProcessedAt": "2026-09-01T15:00:00Z",
  "VideoQuality": null,
  "CalibrationQuality": null
}
```

> **Note:** The worker automatically converts between PascalCase (.NET convention) and snake_case (Python convention).

---

## Docker Deployment

### Build & Run with Docker Compose

```bash
docker-compose up --build
```

This starts the `cheating-detection-worker` container which:
- Connects to RabbitMQ at `host.docker.internal:5672`
- Mounts `./reports` for violation screenshots and `./local_submissions` for video files
- Uses CPU-based PyTorch inference
- Automatically restarts on failure

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RABBITMQ_HOST` | `host.docker.internal` | RabbitMQ server hostname |
| `RABBITMQ_PORT` | `5672` | RabbitMQ port |
| `RABBITMQ_USER` | `admin` | RabbitMQ username |
| `RABBITMQ_PASS` | `admin` | RabbitMQ password |
| `SUBMISSIONS_BASE_PATH` | `/uploads` | Base path for video file resolution |

### Manual Docker Build

```bash
docker build -t proctengine -f dockerfile .
docker run -d \
  --name cheating-detection-worker \
  -e RABBITMQ_HOST=host.docker.internal \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/local_submissions:/uploads \
  proctengine
```

---

## Model Finetuning

The object detection model is fine-tuned from a YOLOv26s base using a custom Roboflow dataset.

### Dataset

- **Source**: [Roboflow Universe — Online Proctoring System](https://universe.roboflow.com/ahmed-mohamed-sifjd/online-proctoring-system-x27ou-nl8r1-intbk/dataset/1)
- **License**: CC BY 4.0
- **Classes**: `book`, `cell phone`, `headphone`, `laptop`, `person`, `tv`
- **Splits**: `train/`, `valid/`, `test/` (images + YOLO-format labels)

### Training Notebook

The full training pipeline is in [`finetune/Finetune_Online_proctoring_YOLO26.ipynb`](finetune/Finetune_Online_proctoring_YOLO26.ipynb). It covers:
1. Dataset download and preparation
2. YOLOv26s base model loading
3. Transfer learning with custom hyperparameters
4. Evaluation metrics and inference examples

---

## API Reference

### `ExamProctor` — High-Level Facade

```python
from pipelines.proctor import ExamProctor

proctor = ExamProctor()
```

#### `proctor.process_video(video_path, student_info=None) → dict`

Analyze a recorded video file.

#### `proctor.process_live(student_info=None, preview=True) → dict`

Start a live webcam proctoring session.

#### Return Value

Both methods return a dictionary:

```python
{
    "risk_score": 42.0,              # 0–100 severity-weighted score
    "suspicious_behavior": "[...]",  # JSON: list of {type, count}
    "detected_objects": "[...]",     # JSON: object violation metadata
    "report_json": "{...}",          # JSON: full report with all violations
    "processed_at": "2026-...",      # ISO-8601 UTC timestamp
    "video_quality": None,           # Reserved for future use
    "calibration_quality": None,     # Reserved for future use
}
```

### Violation Types

| Enum Value | Severity | Trigger |
|------------|----------|---------|
| `FACE_DISAPPEARED` | 1 | Face absent > threshold seconds |
| `GAZE_AWAY` | 2 | Sustained off-screen gaze > threshold seconds |
| `MOUTH_MOVING` | 3 | Consecutive frames of mouth movement |
| `AUDIO_DETECTED` | 3 | Voice/speech detected (live only) |
| `MULTIPLE_FACES` | 4 | More than one face in frame |
| `OBJECT_DETECTED` | 5 | Forbidden object (book, phone, headphone) detected |

---

## Risk Scoring

The risk score is calculated as:

```
risk_score = min(100, Σ severity(violation_type) × 2.0)
```

Each violation contributes its configured severity weight (from `global.severity_levels`). The score is capped at 100. Higher scores indicate more suspicious exam sessions.

**Example:**
- 5 × `GAZE_AWAY` (severity 2) + 1 × `OBJECT_DETECTED` (severity 5) = (10 + 5) × 2 = **30.0**

---

<div align="center">


</div>
