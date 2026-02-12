import cv2
import mediapipe as mp
import numpy as np
from eye_utils import eye_aspect_ratio
from mouth_utils import is_yawning
from alert import AlertSystem   # if file name is alert_system.py
from config import EAR_THRESHOLD, EAR_CONSEC_FRAMES, YAWN_CONSEC_FRAMES

alert = AlertSystem("sounds/alarm.wav")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

eye_counter = 0
yawn_counter = 0

def process_frame(frame):
    global eye_counter, yawn_counter

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) 
                    for i in [33, 160, 158, 133, 153, 144]]
        right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) 
                     for i in [362, 385, 387, 263, 373, 380]]

        ear = (eye_aspect_ratio(np.array(left_eye)) +
               eye_aspect_ratio(np.array(right_eye))) / 2.0

        if ear < EAR_THRESHOLD:
            eye_counter += 1
        else:
            eye_counter = 0

        x1, y1 = int(landmarks[78].x * w), int(landmarks[78].y * h)
        x2, y2 = int(landmarks[308].x * w), int(landmarks[308].y * h)
        mouth_roi = frame[y1:y2, x1:x2]

        if mouth_roi.size > 0 and is_yawning(mouth_roi):
            yawn_counter += 1
        else:
            yawn_counter = 0

        if eye_counter >= EAR_CONSEC_FRAMES or yawn_counter >= YAWN_CONSEC_FRAMES:
            cv2.putText(frame, "DROWSY!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            alert.play()
        else:
            cv2.putText(frame, "ALERT", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
