import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import imutils
import os
import sys
import winsound

# 1. Deployment Path Logic (PyInstaller temporary folder handling)
def resource_path(relative_path):
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# 2. Eye Aspect Ratio (EAR) Function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Parameters
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
counter = 0

# 3. Load Detector and Predictor using resource_path
detector = dlib.get_frontal_face_detector()
predictor_path = resource_path("shape_predictor_68_face_landmarks.dat")

if not os.path.exists(predictor_path):
    print(f"❌ Error: Model file not found at {predictor_path}")
    sys.exit()

predictor = dlib.shape_predictor(predictor_path)

# Landmark indices for eyes
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# Start Video Stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        # Draw visual markers
        leftHull = cv2.convexHull(leftEye)
        rightHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)

        # Alert Logic
        if ear < EAR_THRESHOLD:
            counter += 1
            if counter >= CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(2500, 500)
        else:
            counter = 0

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Driver Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()