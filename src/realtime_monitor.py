"""
Real-Time Student Engagement Monitor (Single Feed)
--------------------------------------------------
Detects a single student's face using Mediapipe + OpenCV
and classifies engagement as:
  - Attentive
  - Confused
  - Distracted
Logs saved automatically in 'logs/' folder.
Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from datetime import datetime
import os
import pandas as pd

# ---------------- CONFIG ----------------
CAMERA_INDEX = 0
EAR_THRESHOLD = 0.20
EAR_CONSEC_FRAMES = 2
YAW_THRESHOLD_DEG = 12.0
BLINK_RATE_HIGH = 0.6
WINDOW_SECONDS = 2.5

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
NOSE_TIP = 1


# ---------- Helper functions ----------
def normalized_to_pixel_coords(x, y, w, h):
    return int(x * w), int(y * h)

def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        x, y = normalized_to_pixel_coords(lm.x, lm.y, img_w, img_h)
        pts.append(np.array([x, y], dtype=np.float32))
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def approx_head_yaw(landmarks, img_w):
    """Approximate head yaw (degrees) using pixel coordinates.

    Uses the horizontal offset of the nose tip from the eye midpoint and
    computes an angle via arctan(delta_x / eye_distance). This produces
    a more meaningful degree value than the previous simple scaling.
    """
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


# ---------- Main ----------
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    os.makedirs("logs", exist_ok=True)
    session_file = f"logs/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_log = pd.DataFrame(columns=["timestamp", "label", "yaw", "blink_rate"])

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        print("✅ Real-Time Engagement Monitor started. Press 'q' to quit.\n")

        window = deque()
        consec_closed = 0
        blink_count = 0
        last_flush = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img_h, img_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            label = "Attentive"

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, img_w, img_h)

                if ear < EAR_THRESHOLD:
                    consec_closed += 1
                else:
                    if consec_closed >= EAR_CONSEC_FRAMES:
                        blink_count += 1
                    consec_closed = 0

                yaw = approx_head_yaw(face_landmarks.landmark, img_w)
                now = time.time()
                window.append((now, blink_count, yaw))
                while window and now - window[0][0] > WINDOW_SECONDS:
                    window.popleft()

                if len(window) > 1:
                    blink_rate = (window[-1][1] - window[0][1]) / (window[-1][0] - window[0][0])
                    avg_yaw = np.mean([abs(w[2]) for w in window])
                else:
                    blink_rate, avg_yaw = 0, 0

                # Classification
                if avg_yaw > YAW_THRESHOLD_DEG * 1.3:
                    label = "Distracted"
                elif avg_yaw > YAW_THRESHOLD_DEG and blink_rate < BLINK_RATE_HIGH / 2:
                    label = "Distracted"
                elif blink_rate > BLINK_RATE_HIGH:
                    label = "Confused"
                else:
                    label = "Attentive"

                color = (0, 255, 0) if label == "Attentive" else ((0, 255, 255) if label == "Confused" else (0, 0, 255))
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=1)
                )
                cv2.putText(frame, f"{label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Yaw: {avg_yaw:.1f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
                cv2.putText(frame, f"Blink rate: {blink_rate:.2f}/s", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Log data
                df_log = pd.concat([df_log, pd.DataFrame([{
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "label": label,
                    "yaw": round(avg_yaw, 2),
                    "blink_rate": round(blink_rate, 2)
                }])], ignore_index=True)

                # Periodically flush to CSV so the running session is visible to the dashboard
                now = time.time()
                if now - last_flush > 1.0:  # Flush every 1 second
                    df_log.to_csv(session_file, index=False)
                    last_flush = now

            cv2.imshow("🎓 Engagement Monitor (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        df_log.to_csv(session_file, index=False)
        print(f"📁 Session saved at: {session_file}")

if __name__ == "__main__":
    main()
