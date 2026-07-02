import argparse
import cv2
import os
from tqdm import tqdm
from collections import Counter
from datetime import datetime, timedelta
from contextlib import ExitStack


from utils.config_loader import load_config
from utils.annotation import annotate_frame
from models import (
    DetectionResult,
    GazeState,
    ViolationEntry,
    ViolationType,
)
from utils.alert_logger import AlertLogger
from utils.screenshot_utils import ViolationCapturer
from .pipeline_utils import build_detectors

FRAME_SKIP = 1
GAZE_FRAME_SKIP = 2


def format_time(seconds):
    """Format seconds into HH:MM:SS.ms string."""
    if seconds is None or seconds < 0:
        seconds = 0
    seconds = min(seconds, 360_000)
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"




def process_video(video_path, config, preview=False, save_output=False, student_info=None):
    """Process a video file through the full detection pipeline.

    Returns
    -------
    dict
        A dictionary with keys ``violations``,
        ``frames_processed``, and ``output_path``.
    """

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0 or video_fps > 240:
        video_fps = 30.0
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames <= 0 or total_frames > 1_000_000_000:
        total_frames = 0
    duration_sec = total_frames / video_fps if total_frames > 0 else 0.0

    video_name = os.path.basename(video_path)
    print(f"\n{'=' * 60}")
    print(f"  Exam Proctoring — Video Analysis")
    print(f"{'=' * 60}")
    print(f"  File     : {video_name}")
    print(f"  Resolution: {video_width}x{video_height} @ {video_fps:.1f} fps")
    print(f"  Duration : {format_time(duration_sec)} ({total_frames} frames)")
    print(f"  Preview  : {'ON' if preview else 'OFF'}")
    print(f"  Save     : {'ON' if save_output else 'OFF'}")
    print(f"{'=' * 60}\n")

    alert_logger = AlertLogger(config)
    violation_capturer = ViolationCapturer(config)
    violations = []

    stack = ExitStack()
    face_detector, gaze_detector, mouth_monitor, multi_face_detector, object_detector = \
        build_detectors(config, alert_logger)

    stack.enter_context(face_detector)
    stack.enter_context(gaze_detector)
    stack.enter_context(mouth_monitor)
    stack.enter_context(multi_face_detector)
    stack.enter_context(object_detector)

    out_writer = None
    output_path = None
    if save_output:
        output_dir = config.video.recording_path
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"analysis_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, video_fps,
                                     (video_width, video_height))

    frame_id = 0
    gaze_counter = 0
    gaze_away_start_frame = None
    gaze_away_threshold_frames = int(config.detection.eyes.gaze_threshold * video_fps)
    face_absent_start_frame = None
    face_absent_threshold_frames = int(config.detection.face.face_absent_threshold * video_fps)
    last_gaze_state = GazeState()

    pbar = tqdm(
        total=total_frames,
        desc="  Analyzing",
        unit="frame",
        bar_format="{l_bar}{bar:30}{r_bar}",
        dynamic_ncols=True,
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            video_time = frame_id / video_fps

            if frame_id % FRAME_SKIP != 0:
                continue

            results = DetectionResult(
                timestamp=format_time(video_time),
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

            video_timestamp = format_time(video_time)

            if not results.face_present:
                if face_absent_start_frame is None:
                    face_absent_start_frame = frame_id
                frames_absent = frame_id - face_absent_start_frame
                if frames_absent >= face_absent_threshold_frames:
                    duration = frames_absent / video_fps
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cap_res = violation_capturer.capture_violation(frame, ViolationType.FACE_DISAPPEARED, ts_str)
                    violations.append(
                        ViolationEntry(
                            type=ViolationType.FACE_DISAPPEARED,
                            timestamp=video_timestamp,
                            metadata={
                                'duration': f'{duration:.1f} seconds',
                                'video_time': video_time,
                                'image_path': cap_res.get('image_path') if cap_res else None
                            }
                        )
                    )
                    face_absent_start_frame = None
            elif results.multiple_faces:
                face_absent_start_frame = None
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cap_res = violation_capturer.capture_violation(frame, ViolationType.MULTIPLE_FACES, ts_str)
                violations.append(
                    ViolationEntry(
                        type=ViolationType.MULTIPLE_FACES,
                        timestamp=video_timestamp,
                        metadata={'video_time': video_time, 'image_path': cap_res.get('image_path') if cap_res else None}
                    )
                )
            elif results.objects_detected:
                face_absent_start_frame = None
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cap_res = violation_capturer.capture_violation(frame, ViolationType.OBJECT_DETECTED, ts_str)
                violations.append(
                    ViolationEntry(
                        type=ViolationType.OBJECT_DETECTED,
                        timestamp=video_timestamp,
                        metadata={'video_time': video_time, 'image_path': cap_res.get('image_path') if cap_res else None}
                    )
                )
            elif gaze_state.gaze.is_away and gaze_state.gaze_conf >= 0.45:
                face_absent_start_frame = None
                if gaze_away_start_frame is None:
                    gaze_away_start_frame = frame_id
                frames_away = frame_id - gaze_away_start_frame
                if frames_away >= gaze_away_threshold_frames:
                    duration = frames_away / video_fps
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cap_res = violation_capturer.capture_violation(frame, ViolationType.GAZE_AWAY, ts_str)
                    violations.append(
                        ViolationEntry(
                            type=ViolationType.GAZE_AWAY,
                            timestamp=video_timestamp,
                            metadata={
                                'duration': f'{duration:.1f} seconds',
                                'video_time': video_time,
                                'image_path': cap_res.get('image_path') if cap_res else None
                            }
                        )
                    )
                    gaze_away_start_frame = None
            else:
                face_absent_start_frame = None
                gaze_away_start_frame = None

            if results.mouth_moving:
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cap_res = violation_capturer.capture_violation(frame, ViolationType.MOUTH_MOVING, ts_str)
                violations.append(
                    ViolationEntry(
                        type=ViolationType.MOUTH_MOVING,
                        timestamp=video_timestamp,
                        metadata={'video_time': video_time, 'image_path': cap_res.get('image_path') if cap_res else None}
                    )
                )

            annotated = annotate_frame(frame, results, gaze_state)

            if out_writer:
                out_writer.write(annotated)

            if preview:
                cv2.imshow('Video Analysis (press Q to quit)', annotated)
                wait_ms = max(1, int(1000 / video_fps))
                if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                    print("\n\n  Playback interrupted by user.")
                    break
            pbar.update(1)
            if frame_id % 30 == 0:
                pbar.set_postfix(
                    time=format_time(video_time),
                    violations=len(violations),
                )

    finally:
        pbar.close()
        stack.close()
        cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()

    return {
        "violations": violations,
        "frames_processed": frame_id,
        "output_path": output_path,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Process a recorded video through the exam proctoring detection pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m video_pipeline recordings/exam_session.mp4
  python -m video_pipeline recordings/exam_session.mp4 --preview
  python -m video_pipeline recordings/exam_session.mp4 --save-output --student-id S042
        """
    )
    parser.add_argument('video', help='Path to the video file to analyze')
    parser.add_argument('--preview', action='store_true',
                        help='Show annotated video playback during processing')
    parser.add_argument('--save-output', action='store_true',
                        help='Save the annotated video to the recordings directory')
    parser.add_argument('--student-id', default=None,
                        help='Student ID for the report (default: VIDEO_ANALYSIS)')
    parser.add_argument('--student-name', default=None,
                        help='Student name for the report')

    args = parser.parse_args()

    config = load_config()

    student_info = None
    if args.student_id or args.student_name:
        student_info = {
            'id': args.student_id or 'VIDEO_ANALYSIS',
            'name': args.student_name or 'Unknown Student',
            'exam': f'Video: {os.path.basename(args.video)}',
            'course': 'N/A',
        }

    result = process_video(
        video_path=args.video,
        config=config,
        preview=args.preview,
        save_output=args.save_output,
        student_info=student_info,
    )

    violations = result["violations"]
    frames_processed = result["frames_processed"]
    output_path = result["output_path"]

    print(f"\n\n{'=' * 60}")
    print(f"  Analysis Complete")
    print(f"{'=' * 60}")
    print(f"  Frames processed : {frames_processed}")
    print(f"  Total violations : {len(violations)}")

    counts = Counter(v.type.value for v in violations)
    for vtype, count in counts.most_common():
        print(f"    {vtype:25s} : {count}")

    if output_path:
        print(f"\n  Annotated video  : {output_path}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
