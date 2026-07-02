import cv2
from datetime import datetime
from contextlib import ExitStack

from utils.config_loader import load_config
from utils.annotation import annotate_frame
from models import (
    DetectionResult,
    GazeDirection,
    GazeState,
    ViolationEntry,
    ViolationType,
)
from detection.audio_detection import AudioMonitor
from utils.video_utils import VideoRecorder
from utils.alert_logger import AlertLogger
from utils.alert_system import AlertSystem
from utils.screenshot_utils import ViolationCapturer
from pipeline_utils import build_detectors

FRAME_SKIP = 2
GAZE_FRAME_SKIP = 2


def _handle_violation(
    frame,
    violation_type: ViolationType,
    alert_system: AlertSystem,
    violation_capturer: ViolationCapturer,
    violations: list[ViolationEntry],
    results: DetectionResult,
    gaze_state: GazeState,
    extra_metadata: dict | None = None,
):
    """Centralised violation handling: speak alert, capture screenshot, log."""
    alert_system.speak_alert(violation_type)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cap_res = violation_capturer.capture_violation(frame, violation_type, timestamp)

    metadata = extra_metadata or {}
    metadata.update({
        "image_path": cap_res.get("image_path") if cap_res else None,
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


def main():
    config = load_config()
    alert_logger = AlertLogger(config)
    alert_system = AlertSystem(config)
    violation_capturer = ViolationCapturer(config)
    violations = []

    # Placeholder student info — will be replaced with CLI args or user prompt later
    student_info = {
        "id": "STUDENT_001",
        "name": "John Doe",
        "exam": "Final Examination",
        "course": "Computer Science 101",
    }

    video_recorder = VideoRecorder(config)

    audio_monitor = AudioMonitor(
        config,
        alert_system=alert_system,
        alert_logger=alert_logger,
    )

    if config.detection.audio_monitoring.enabled:
        audio_monitor.start()

    try:
        stack = ExitStack()
        face_detector, gaze_detector, mouth_monitor, multi_face_detector, object_detector = \
            build_detectors(config, alert_logger)

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
        last_annotated_frame = None
        frame_id = 0
        gaze_counter = 0
        gaze_away_start = None
        face_absent_start = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            if frame_id % FRAME_SKIP != 0:
                frame_id += 1
                if last_annotated_frame is not None:
                    cv2.imshow('Exam Proctoring', last_annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
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
                        frame, ViolationType.FACE_DISAPPEARED,
                        alert_system, violation_capturer, violations,
                        results, gaze_state,
                        {"duration": f"{face_absent_duration:.1f} seconds"},
                    )
                    face_absent_start = None
            elif results.multiple_faces:
                face_absent_start = None
                _handle_violation(
                    frame, ViolationType.MULTIPLE_FACES,
                    alert_system, violation_capturer, violations,
                    results, gaze_state,
                    None,
                )
            elif results.objects_detected:
                face_absent_start = None
                _handle_violation(
                    frame, ViolationType.OBJECT_DETECTED,
                    alert_system, violation_capturer, violations,
                    results, gaze_state,
                    None,
                )
            elif gaze_state.gaze.is_away and gaze_state.gaze_conf >= 0.45:
                face_absent_start = None

                if gaze_away_start is None:
                    gaze_away_start = datetime.now()

                gaze_away_duration = (datetime.now() - gaze_away_start).total_seconds()
                if gaze_away_duration >= config.detection.eyes.gaze_threshold:
                    _handle_violation(
                        frame, ViolationType.GAZE_AWAY,
                        alert_system, violation_capturer, violations,
                        results, gaze_state,
                        {"duration": f"{gaze_away_duration:.1f} seconds"},
                    )
                    gaze_away_start = None
            else:
                gaze_away_start = None
                face_absent_start = None

            if results.mouth_moving:
                _handle_violation(
                    frame, ViolationType.MOUTH_MOVING,
                    alert_system, violation_capturer, violations,
                    results, gaze_state,
                    {"duration": "5+ seconds"},
                )

            annotated = annotate_frame(frame, results, gaze_state)
            last_annotated_frame = annotated
            video_recorder.record_frame(annotated)

            cv2.imshow('Exam Proctoring', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_id += 1

    finally:
        # Stop audio monitor if active
        if config.detection.audio_monitoring.enabled and 'audio_monitor' in locals():
            try:
                audio_monitor.stop()
            except Exception:
                pass

        if 'stack' in locals():
            stack.close()

        print("Exam session ended.")

        video_data = video_recorder.stop_recording()
        print(f"Webcam recording saved: {video_data.filename}")

        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()