"""
RabbitMQ consumer worker for cheating-detection requests.

Listens on ``cheating-detection-queue``, runs each video through the
detection pipeline, and publishes the result to
``cheating-detection-result-queue``.

Run with::

    cd src
    python -m messaging.rabbitmq_worker
"""

import sys
import os
import json
import time
import traceback
from datetime import timezone, datetime
from dataclasses import asdict, fields

import pika

from models.messaging import CheatingDetectionRequest, CheatingDetectionResult
from pipelines.proctor import ExamProctor

Config = None


def get_config():
    global Config
    if Config is None:
        from utils.config_loader import load_config
        Config = load_config().rabbitmq
    return Config

def process_message(body: bytes) -> dict:
    print(f" [>] process_message started at {datetime.now(timezone.utc).isoformat()}")

    try:
        raw_data = json.loads(body.decode("utf-8"))
        print(f" [>] Message parsed successfully. Top-level keys: {list(raw_data.keys())}")

        request = CheatingDetectionRequest(
            quiz_submission_id=raw_data.get("QuizSubmissionId", raw_data.get("quiz_submission_id", "")),
            quiz_id=raw_data.get("QuizId", raw_data.get("quiz_id", "")),
            student_user_id=raw_data.get("StudentUserId", raw_data.get("student_user_id", "")),
            user_full_name=raw_data.get("UserFullName", raw_data.get("user_full_name", "")),
            student_id_in_course=raw_data.get("StudentIdInCourse", raw_data.get("student_id_in_course")),
            video_path=raw_data.get("VideoPath", raw_data.get("video_path", "")),
            video_size=raw_data.get("VideoSize", raw_data.get("video_size", 0)),
            video_duration=raw_data.get("VideoDuration", raw_data.get("video_duration", "")),
            submitted_at=raw_data.get("SubmittedAt", raw_data.get("submitted_at", "")),
        )
        video_path = request.video_path
        config = get_config()
        if not os.path.isabs(video_path) and config.submissions_base_path:
            video_path = os.path.join(config.submissions_base_path, video_path)

        print(f" [>] Processing video: '{video_path}' for student '{request.user_full_name}'")

        student_info = {
            "id": request.student_user_id or "UNKNOWN_ID",
            "name": request.user_full_name or "Unknown Student",
            "exam": f"Quiz: {request.quiz_id}" if request.quiz_id else "Automated Analysis",
            "course": "N/A",
        }

        proctor = ExamProctor()
        proctor_result = proctor.process_video(video_path, student_info=student_info)

        result = CheatingDetectionResult(
            quiz_submission_id=request.quiz_submission_id,
            risk_score=proctor_result["risk_score"],
            suspicious_behavior=proctor_result["suspicious_behavior"],
            detected_objects=proctor_result["detected_objects"],
            report_json=proctor_result["report_json"],
            processed_at=proctor_result["processed_at"],
            video_quality=proctor_result.get("video_quality"),
            calibration_quality=proctor_result.get("calibration_quality"),
        )

        return asdict(result)

    except Exception as e:
        print(f" [!] Error during message processing: {type(e).__name__}: {e}")
        print(f" [!] Traceback:\n{traceback.format_exc()}")
        raise


def connect_rabbitmq():
    retries = 10
    delay = 5
    config = get_config()

    for i in range(retries):
        try:
            print(
                f" [>] Connecting to RabbitMQ at {config.host}:{config.port} "
                f"vhost='{config.vhost}'"
            )
            credentials = pika.PlainCredentials(config.user, config.password)
            parameters = pika.ConnectionParameters(
                host=config.host,
                port=config.port,
                virtual_host=config.vhost,
                credentials=credentials,
                connection_attempts=3,
                retry_delay=2,
            )
            connection = pika.BlockingConnection(parameters=parameters)
            print(" [x] Connected to RabbitMQ successfully")
            return connection
        except pika.exceptions.AMQPConnectionError as e:
            print(
                f" [!] RabbitMQ connection failed (attempt {i + 1}/{retries}). "
                f"Retrying in {delay}s: {e}"
            )
            time.sleep(delay)

    raise RuntimeError("Failed to connect to RabbitMQ after all retries.")

def main():
    config = get_config()
    print(f"[*] Worker starting up at {datetime.now(timezone.utc).isoformat()}")
    print(
        f"[*] Config - REQUEST_QUEUE='{config.request_queue}', "
        f"RESPONSE_QUEUE='{config.response_queue}', "
        f"BASE_PATH='{config.submissions_base_path}'"
    )

    connection = connect_rabbitmq()
    channel = connection.channel()

    channel.queue_declare(queue=config.request_queue, durable=True)
    channel.queue_declare(queue=config.response_queue, durable=True)
    print(
        f" [*] Queues declared. Waiting for requests on "
        f"'{config.request_queue}'. To exit press CTRL+C"
    )

    def callback(ch, method, properties, body):
        print(f"\n{'=' * 60}")
        print(
            f" [x] Received message - size={len(body)} bytes, "
            f"timestamp={datetime.now(timezone.utc).isoformat()}"
        )

        try:
            response_payload = process_message(body)

            pascal_payload = {
                "QuizSubmissionId": response_payload.get("quiz_submission_id", ""),
                "RiskScore": response_payload.get("risk_score", 0.0),
                "SuspiciousBehavior": response_payload.get("suspicious_behavior", ""),
                "DetectedObjects": response_payload.get("detected_objects", ""),
                "ReportJson": response_payload.get("report_json", ""),
                "ProcessedAt": response_payload.get("processed_at", ""),
                "VideoQuality": response_payload.get("video_quality"),
                "CalibrationQuality": response_payload.get("calibration_quality"),
            }

            ch.basic_publish(
                exchange="",
                routing_key=config.response_queue,
                body=json.dumps(pascal_payload),
                properties=pika.BasicProperties(delivery_mode=2),
            )

            print(
                f" [x] Response published to '{config.response_queue}' — "
                f"QuizSubmissionId={response_payload['quiz_submission_id']}, "
                f"RiskScore={response_payload['risk_score']}, "
                f"ProcessedAt={response_payload['processed_at']}"
            )
        except Exception as e:
            print(
                f" [!] Unexpected error in callback (outside process_message): "
                f"{type(e).__name__}: {e}"
            )
            print(f" [!] Traceback:\n{traceback.format_exc()}")
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            print(f" [>] Message acknowledged (delivery_tag={method.delivery_tag})")
            print(f"{'=' * 60}\n")

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=config.request_queue, on_message_callback=callback)

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\n[*] Stopping consumer...")
        channel.stop_consuming()
    finally:
        connection.close()
        print("[*] Connection closed")


if __name__ == "__main__":
    main()
