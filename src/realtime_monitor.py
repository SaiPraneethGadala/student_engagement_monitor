"""
Real-Time Student Engagement Monitor (Single Feed)
--------------------------------------------------
Uses Mediapipe FaceMesh when available. If Mediapipe `solutions`
API is unavailable (common on newer Python builds), falls back to
OpenCV face detection so logging/dashboard still work.

Press 'q' to quit.
"""

import os
import signal
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

try:
    import mediapipe as mp
except Exception:
    mp = None

# ---------------- CONFIG ----------------
CAMERA_INDEX = 0
EAR_THRESHOLD = 0.20
EAR_CONSEC_FRAMES = 2
YAW_THRESHOLD_DEG = 12.0
BLINK_RATE_HIGH = 0.6
WINDOW_SECONDS = 2.5
SLEEP_EYE_CLOSED_SECONDS = 2.2

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
NOSE_TIP = 1

HAS_MP_SOLUTIONS = bool(mp is not None and hasattr(mp, "solutions"))
if HAS_MP_SOLUTIONS:
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
else:
    mp_face_mesh = None
    mp_drawing = None

STOP_REQUESTED = False


def request_stop(signum=None, frame=None):
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def should_stop():
    return STOP_REQUESTED


def normalized_to_pixel_coords(x, y, w, h):
    return int(x * w), int(y * h)


def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        x, y = normalized_to_pixel_coords(lm.x, lm.y, img_w, img_h)
        pts.append(np.array([x, y], dtype=np.float32))
    a = np.linalg.norm(pts[1] - pts[5])
    b = np.linalg.norm(pts[2] - pts[4])
    c = np.linalg.norm(pts[0] - pts[3])
    if c == 0:
        return 0.0
    return (a + b) / (2.0 * c)


def approx_head_yaw_from_landmarks(landmarks, img_w):
    left_eye_x = landmarks[LEFT_EYE[0]].x * img_w
    right_eye_x = landmarks[RIGHT_EYE[0]].x * img_w
    nose_x = landmarks[NOSE_TIP].x * img_w
    eye_mid = (left_eye_x + right_eye_x) / 2.0
    delta = nose_x - eye_mid
    eye_dist = right_eye_x - left_eye_x
    if eye_dist == 0:
        return 0.0
    yaw_rad = np.arctan2(delta, eye_dist)
    return abs(np.degrees(yaw_rad))


def classify(avg_yaw, blink_rate, yaw_jitter=0.0, face_visible=True, sleeping=False):
    if sleeping:
        return "Sleeping"
    if not face_visible:
        return "Distracted"
    if avg_yaw > YAW_THRESHOLD_DEG * 1.3:
        return "Distracted"
    if avg_yaw > YAW_THRESHOLD_DEG and blink_rate < BLINK_RATE_HIGH / 2:
        return "Distracted"
    # Lower confusion threshold and allow confusion from unstable head movement.
    if blink_rate > 0.35 or (yaw_jitter > 4.0 and avg_yaw < (YAW_THRESHOLD_DEG + 3.0)):
        return "Confused"
    return "Attentive"


def color_for(label):
    if label == "Attentive":
        return (0, 255, 0)
    if label == "Confused":
        return (0, 255, 255)
    if label == "Sleeping":
        return (255, 120, 0)
    return (0, 0, 255)


def run_with_mediapipe(cap, session_file):
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        print("[INFO] Using Mediapipe FaceMesh backend.")
        print("[INFO] Real-Time Engagement Monitor started. Press 'q' to quit.")

        df_log = pd.DataFrame(columns=["timestamp", "label", "yaw", "blink_rate"])
        window = deque()
        consec_closed = 0
        eyes_closed_since = None
        blink_count = 0
        last_flush = time.time()

        while True:
            if should_stop():
                break
            ret, frame = cap.read()
            if not ret:
                break

            img_h, img_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            label = "Attentive"
            avg_yaw = 0.0
            blink_rate = 0.0
            yaw_jitter = 0.0

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, img_w, img_h)
                now = time.time()

                if ear < EAR_THRESHOLD:
                    consec_closed += 1
                    if eyes_closed_since is None:
                        eyes_closed_since = now
                else:
                    if consec_closed >= EAR_CONSEC_FRAMES:
                        blink_count += 1
                    consec_closed = 0
                    eyes_closed_since = None

                yaw = approx_head_yaw_from_landmarks(face_landmarks.landmark, img_w)
                window.append((now, blink_count, yaw))
                while window and now - window[0][0] > WINDOW_SECONDS:
                    window.popleft()

                if len(window) > 1:
                    blink_rate = (window[-1][1] - window[0][1]) / max((window[-1][0] - window[0][0]), 1e-6)
                    yaw_values = [abs(w[2]) for w in window]
                    avg_yaw = float(np.mean(yaw_values))
                    yaw_jitter = float(np.std(yaw_values))

                sleeping = bool(eyes_closed_since is not None and (now - eyes_closed_since) >= SLEEP_EYE_CLOSED_SECONDS)
                label = classify(avg_yaw, blink_rate, yaw_jitter=yaw_jitter, face_visible=True, sleeping=sleeping)
                color = color_for(label)

                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=1),
                )

            color = color_for(label)
            cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Yaw: {avg_yaw:.1f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
            cv2.putText(frame, f"Blink rate: {blink_rate:.2f}/s", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Yaw jitter: {yaw_jitter:.2f}", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"Sleep eyes-closed threshold: {SLEEP_EYE_CLOSED_SECONDS:.1f}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

            df_log = pd.concat(
                [
                    df_log,
                    pd.DataFrame(
                        [
                            {
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "label": label,
                                "yaw": round(avg_yaw, 2),
                                "blink_rate": round(blink_rate, 2),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

            now = time.time()
            if now - last_flush > 1.0:
                df_log.to_csv(session_file, index=False)
                last_flush = now

            cv2.imshow("Engagement Monitor (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            try:
                if cv2.getWindowProperty("Engagement Monitor (q to quit)", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                pass

        return df_log


def run_with_opencv_fallback(cap, session_file):
    print("[WARN] Mediapipe 'solutions' API is unavailable in this Python environment.")
    print("[INFO] Falling back to OpenCV face tracking backend.")
    print("[INFO] Real-Time Engagement Monitor started. Press 'q' to quit.")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    df_log = pd.DataFrame(columns=["timestamp", "label", "yaw", "blink_rate"])
    window = deque()
    face_state_window = deque()
    last_flush = time.time()
    last_face_center_x = None
    eyes_closed_since = None

    while True:
        if should_stop():
            break
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        blink_rate = 0.0
        face_visible = len(faces) > 0
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            face_center_x = x + w / 2.0
            yaw = abs((face_center_x - (img_w / 2.0)) / max(img_w / 2.0, 1.0)) * 30.0
            face_roi_gray = gray[y : y + h, x : x + w]
            eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=6, minSize=(18, 18))
            eyes_visible = len(eyes) >= 1
            now = time.time()
            if eyes_visible:
                eyes_closed_since = None
            elif eyes_closed_since is None:
                eyes_closed_since = now
            if last_face_center_x is None:
                center_shift = 0.0
            else:
                center_shift = abs(face_center_x - last_face_center_x) / max(img_w, 1.0)
            last_face_center_x = face_center_x
            cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 220, 255), 2)
        else:
            yaw = 25.0
            center_shift = 0.0
            last_face_center_x = None
            eyes_closed_since = None

        now = time.time()
        window.append((now, yaw))
        face_state_window.append((now, face_visible))
        while window and now - window[0][0] > WINDOW_SECONDS:
            window.popleft()
        while face_state_window and now - face_state_window[0][0] > WINDOW_SECONDS:
            face_state_window.popleft()

        yaw_values = [w[1] for w in window] if window else [yaw]
        avg_yaw = float(np.mean(yaw_values))
        yaw_jitter = float(np.std(yaw_values))

        # In fallback mode we approximate confusion via rapid movement and
        # frequent face visibility changes (detector loses/reacquires face).
        if len(face_state_window) > 1:
            toggles = sum(
                1
                for i in range(1, len(face_state_window))
                if face_state_window[i][1] != face_state_window[i - 1][1]
            )
            span = max(face_state_window[-1][0] - face_state_window[0][0], 1e-6)
            blink_rate = toggles / span
        blink_rate += center_shift * 2.0

        sleeping = bool(face_visible and eyes_closed_since is not None and (now - eyes_closed_since) >= SLEEP_EYE_CLOSED_SECONDS)
        label = classify(avg_yaw, blink_rate, yaw_jitter=yaw_jitter, face_visible=face_visible, sleeping=sleeping)
        color = color_for(label)

        cv2.putText(frame, f"{label} (fallback)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Yaw: {avg_yaw:.1f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
        cv2.putText(frame, f"Blink rate: {blink_rate:.2f}/s", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Yaw jitter: {yaw_jitter:.2f}", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        df_log = pd.concat(
            [
                df_log,
                pd.DataFrame(
                    [
                        {
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "label": label,
                            "yaw": round(avg_yaw, 2),
                            "blink_rate": round(blink_rate, 2),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        now = time.time()
        if now - last_flush > 1.0:
            df_log.to_csv(session_file, index=False)
            last_flush = now

        cv2.imshow("Engagement Monitor (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        try:
            if cv2.getWindowProperty("Engagement Monitor (q to quit)", cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            pass

    return df_log


def main():
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    session_file = os.path.join(logs_dir, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    try:
        if HAS_MP_SOLUTIONS:
            df_log = run_with_mediapipe(cap, session_file)
        else:
            df_log = run_with_opencv_fallback(cap, session_file)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    df_log.to_csv(session_file, index=False)
    print(f"[INFO] Session saved at: {session_file}")


if __name__ == "__main__":
    main()
