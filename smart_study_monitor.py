"""
Smart Study Monitor — recreation inspired by the Instagram Reel.
Features:
- Webcam face presence check
- Eye-closure based "WAKE UP & STUDY!" alert (approximate)
- YOLOv8 phone detection for "PUT THE PHONE AWAY!"
- Face-not-visible alert for "DON'T COVER YOUR FACE!"
- Optional alert sounds (Windows beep fallback)

Install:
    pip install opencv-python mediapipe ultralytics pygame

Run:
    python smart_study_monitor.py

Notes:
- First run of YOLOv8 may download yolov8n.pt automatically.
- Camera-based detections are estimates and may produce false alerts.
- Press Q to quit.
"""
import time
import threading
import os
import cv2
import mediapipe as mp
from ultralytics import YOLO

# Optional sound support. If pygame/audio isn't available, visual alerts still work.
try:
    import pygame
    pygame.mixer.init()
    SOUND_AVAILABLE = True
except Exception:
    pygame = None
    SOUND_AVAILABLE = False

CAMERA_INDEX = 0
PHONE_CONFIDENCE = 0.35
EYES_CLOSED_SECONDS = 2.0
FACE_MISSING_SECONDS = 2.0
PHONE_ALERT_COOLDOWN = 3.0
SOUND_COOLDOWN = 4.0

# MediaPipe Face Mesh eye landmarks (indices around each eye).
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def eye_aspect_ratio(landmarks, eye_indices, width, height):
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        points.append((int(lm.x * width), int(lm.y * height)))
    # EAR = average vertical distance / horizontal distance
    vertical_1 = distance(points[1], points[5])
    vertical_2 = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])
    if horizontal == 0:
        return 1.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def draw_alert(frame, message, color=(0, 0, 255)):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(message, font, scale, thickness)
    x = max(10, (w - tw) // 2)
    y = 48
    cv2.rectangle(frame, (x - 12, y - th - 12), (x + tw + 12, y + 12), color, -1)
    cv2.putText(frame, message, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def play_alert(message):
    # Play the matching spoken alert from the audio folder.
    audio_files = {
        "WAKE UP & STUDY!": "wake_up_study.wav",
        "DON'T COVER YOUR FACE!": "dont_cover_face.wav",
        "PUT THE PHONE AWAY!": "put_phone_away.wav",
    }
    filename = audio_files.get(message)
    audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio", filename) if filename else None

    try:
        if pygame is not None and audio_path and os.path.isfile(audio_path):
            sound = pygame.mixer.Sound(audio_path)
            sound.play()
            # Wait for this short spoken alert to finish before this thread exits.
            while pygame.mixer.get_busy():
                time.sleep(0.05)
        else:
            import winsound
            winsound.Beep(1000, 350)
    except Exception:
        try:
            import winsound
            winsound.Beep(1000, 350)
        except Exception:
            pass


def main():
    print("Loading YOLOv8 model (the first run may download model weights)...")
    model = YOLO("yolov8n.pt")  # COCO class 67 = cell phone
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check CAMERA_INDEX or camera permissions.")

    eyes_closed_since = None
    face_missing_since = None
    last_phone_alert = 0.0
    last_sound = 0.0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_result = face_mesh.process(rgb)
            alert = None
            now = time.time()

            face_detected = bool(face_result.multi_face_landmarks)
            if not face_detected:
                if face_missing_since is None:
                    face_missing_since = now
                elif now - face_missing_since >= FACE_MISSING_SECONDS:
                    alert = "DON'T COVER YOUR FACE!"
                eyes_closed = False
                eyes_closed_since = None
            else:
                face_missing_since = None
                face_landmarks = face_result.multi_face_landmarks[0].landmark
                left_ear = eye_aspect_ratio(face_landmarks, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE, w, h)
                ear = (left_ear + right_ear) / 2.0

                # Approximate threshold; adjust for camera angle and lighting.
                eyes_closed = ear < 0.19
                cv2.putText(frame, f"Eye ratio: {ear:.2f}", (12, h - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if eyes_closed:
                    if eyes_closed_since is None:
                        eyes_closed_since = now
                    elif now - eyes_closed_since >= EYES_CLOSED_SECONDS:
                        alert = "WAKE UP & STUDY!"
                else:
                    eyes_closed_since = None

                # Draw a small face status label
                cv2.putText(frame, "Face detected", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

            # Detect cell phones using the COCO-trained YOLO model.
            # Class ID 67 is "cell phone" in the standard COCO label set.
            try:
                detections = model.predict(frame, imgsz=640, conf=PHONE_CONFIDENCE,
                                            classes=[67], verbose=False)
                phone_found = False
                for result in detections:
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        phone_found = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(frame, f"Phone {conf:.0%}", (x1, max(25, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                if phone_found and now - last_phone_alert >= PHONE_ALERT_COOLDOWN:
                    alert = "PUT THE PHONE AWAY!"
                    last_phone_alert = now
            except Exception as exc:
                cv2.putText(frame, f"Phone detection issue: {str(exc)[:50]}", (12, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

            if alert:
                draw_alert(frame, alert, (180, 0, 180) if "PHONE" in alert else (0, 0, 220))
                if now - last_sound >= SOUND_COOLDOWN:
                    threading.Thread(target=play_alert, args=(alert,), daemon=True).start()
                    last_sound = now

            cv2.putText(frame, "SMART STUDY MONITOR  |  Press Q to quit",
                        (12, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow("Smart Study Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
