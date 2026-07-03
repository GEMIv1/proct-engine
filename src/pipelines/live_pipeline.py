import cv2
from datetime import datetime, timezone
from contextlib import ExitStack

import numpy as np

from utils.config_loader import load_config
from utils.annotation import annotate_frame
from models import (
    DetectionResult,
    GazeState,
    ViolationEntry,
    ViolationType,
)
from detection.audio_detection import AudioMonitor
from utils.video_utils import VideoRecorder
from utils.alert_system import AlertSystem
from utils.pipeline_utils import build_detectors
from utils.screenshot_capture import ScreenshotCapture

FRAME_SKIP = 2
GAZE_FRAME_SKIP = 2


def _handle_violation(
    violation_type: ViolationType,
    alert_system: AlertSystem,
    violations: list[ViolationEntry],
    results: DetectionResult,
    gaze_state: GazeState,
    frame: np.ndarray,
    screenshot_capture: ScreenshotCapture,
    extra_metadata: dict | None = None,
):
    """Centralised violation handling: speak alert, capture screenshot, log."""
    alert_system.speak_alert(violation_type)
    timestamp = datetime.now(timezone.utc).isoformat()

    # Capture an annotated screenshot of the violation frame
    image_path = screenshot_capture.capture(frame, violation_type.value, timestamp)

    metadata = extra_metadata or {}
    metadata["image_path"] = image_path
    metadata.update({
        "frame": {
            "face_present": results.face_present,
            "gaze_direction": results.gaze_direction.value,
            "mouth_moving": results.mouth_moving,
            "multiple_faces": results.multiple_faces,
            "objects_detected": results.objects_detected,
        },
        "gaze_state": {
            "gaze": gaze_state.gaze.value,
            "gaze_conf": gaze_state.gaze_conf,
        },
    })
    violations.append(
        ViolationEntry(
            type=violation_type,
            timestamp=timestamp,
            metadata=metadata,
        )
    )


def process_live(config=None, student_info=None, preview=True):
    if config is None:
        config = load_config()
    alert_system = AlertSystem(config)
    screenshot_capture = ScreenshotCapture(cooldown=config.alert.cooldown)
    violations = []

    video_recorder = VideoRecorder(config)

    audio_monitor = AudioMonitor(
        config,
        alert_system=alert_system,
    )

    if config.detection.audio_monitoring.enabled:
        audio_monitor.start()

    frame_id = 0
    try:
        stack = ExitStack()
        face_detector, gaze_detector, mouth_monitor, multi_face_detector, object_detector = \
            build_detectors(config)

        stack.enter_context(face_detector)
        stack.enter_context(gaze_detector)
        stack.enter_context(mouth_monitor)
        stack.enter_context(multi_face_detector)
        stack.enter_context(object_detector)

        video_recorder.start_recording()
        cap = cv2.VideoCapture(config.video.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.video.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.video.resolution[1])

        last_gaze_state = GazeState()
        gaze_counter = 0
        gaze_away_start = None
        face_absent_start = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            frame = cv2.flip(frame, 1)

            if frame_id % FRAME_SKIP != 0:
                continue

            results = DetectionResult(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            results.face_present = face_detector.process(frame)

            if gaze_counter % GAZE_FRAME_SKIP == 0:
                gaze_state = gaze_detector.process(frame)
                last_gaze_state = gaze_state
            else:
                gaze_state = last_gaze_state
            gaze_counter += 1

            results.gaze_direction = gaze_state.gaze
            results.gaze_conf = gaze_state.gaze_conf

            results.mouth_moving = mouth_monitor.process(frame)
            results.multiple_faces = multi_face_detector.process(frame)
            results.objects_detected = object_detector.process(frame)

            if not results.face_present:
                if face_absent_start is None:
                    face_absent_start = datetime.now()

                face_absent_duration = (datetime.now() - face_absent_start).total_seconds()
                if face_absent_duration >= config.detection.face.face_absent_threshold:
                    _handle_violation(
                        ViolationType.FACE_DISAPPEARED,
                        alert_system, violations,
                        results, gaze_state,
                        frame, screenshot_capture,
                        {"duration": f"{face_absent_duration:.1f} seconds"},
                    )
                    face_absent_start = None
            elif results.multiple_faces:
                face_absent_start = None
                _handle_violation(
                    ViolationType.MULTIPLE_FACES,
                    alert_system, violations,
                    results, gaze_state,
                    frame, screenshot_capture,
                )
            elif results.objects_detected:
                face_absent_start = None
                _handle_violation(
                    ViolationType.OBJECT_DETECTED,
                    alert_system, violations,
                    results, gaze_state,
                    frame, screenshot_capture,
                )
            elif gaze_state.gaze.is_away and gaze_state.gaze_conf >= 0.45:
                face_absent_start = None

                if gaze_away_start is None:
                    gaze_away_start = datetime.now()

                gaze_away_duration = (datetime.now() - gaze_away_start).total_seconds()
                if gaze_away_duration >= config.detection.eyes.gaze_threshold:
                    _handle_violation(
                        ViolationType.GAZE_AWAY,
                        alert_system, violations,
                        results, gaze_state,
                        frame, screenshot_capture,
                        {"duration": f"{gaze_away_duration:.1f} seconds"},
                    )
                    gaze_away_start = None
            else:
                gaze_away_start = None
                face_absent_start = None

            if results.mouth_moving:
                _handle_violation(
                    ViolationType.MOUTH_MOVING,
                    alert_system, violations,
                    results, gaze_state,
                    frame, screenshot_capture,
                )

            annotated = annotate_frame(frame, results, gaze_state)
            video_recorder.record_frame(annotated)

            if preview:
                cv2.imshow('Exam Proctoring', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_id += 1

    finally:
        # Stop audio monitor if active
        if config.detection.audio_monitoring.enabled and 'audio_monitor' in locals() and audio_monitor is not None:
            try:
                audio_monitor.stop()
            except Exception:
                pass

        if 'stack' in locals():
            stack.close()

        print("Exam session ended.")

        video_data = video_recorder.stop_recording()
        if video_data:
            print(f"Webcam recording saved: {video_data.filename}")

        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if preview:
            cv2.destroyAllWindows()

    return {
        "violations": violations,
        "frames_processed": frame_id,
        "output_path": video_data.filename if video_data else None,
    }


if __name__ == '__main__':
    process_live(preview=True)