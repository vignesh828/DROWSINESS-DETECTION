import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import os
import base64

# 1. Sound Logic for Browser (HTML5 Injection)
def get_audio_html(audio_file):
    if os.path.exists(audio_file):
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode()
            return f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}">'
    return ""

# 2. Eye Aspect Ratio (EAR) Logic
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# 3. Video Processing Class
class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        # Ensure the .dat file is in your GitHub root folder
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.counter = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        alert_status = False

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])
            
            leftEAR = eye_aspect_ratio(shape[42:48])
            rightEAR = eye_aspect_ratio(shape[36:42])
            ear = (leftEAR + rightEAR) / 2.0

            # Draw contours on eyes
            leftHull = cv2.convexHull(shape[42:48])
            rightHull = cv2.convexHull(shape[36:42])
            cv2.drawContours(img, [leftHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightHull], -1, (0, 255, 0), 1)

            if ear < 0.25:
                self.counter += 1
                if self.counter >= 20:
                    cv2.putText(img, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    alert_status = True
            else:
                self.counter = 0

            cv2.putText(img, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return img

# Streamlit UI
st.set_page_config(page_title="AI Driver Safety Monitor", layout="centered")
st.title("AI Driver Safety Monitor 🚗💤")
st.write("Mechanical Engineering Project — IIT Kharagpur")

# Sidebar info
st.sidebar.title("About Project")
st.sidebar.info("This system uses Computer Vision to detect driver fatigue in real-time. Created by Buddha Vignesh.")

# Start Streamer
ctx = webrtc_streamer(key="drowsiness-det", video_transformer_factory=DrowsinessTransformer)

# Sound Alert Injection (if drowsy)
if ctx.video_transformer and ctx.video_transformer.counter >= 20:
    st.markdown(get_audio_html("Amelia Island.mp3"), unsafe_allow_html=True)
    st.error("WAKE UP! Drowsiness Detected.")