import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import os
import base64

# 1. HTML5 Sound Injection
def get_audio_html(audio_file):
    if os.path.exists(audio_file):
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode()
            return f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}">'
    return ""

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.counter = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        
        drowsy_detected = False

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])
            ear = (eye_aspect_ratio(shape[42:48]) + eye_aspect_ratio(shape[36:42])) / 2.0

            if ear < 0.25:
                self.counter += 1
                if self.counter >= 20:
                    cv2.putText(img, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # We set a flag here
            else:
                self.counter = 0

            cv2.putText(img, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return img

# --- UI ---
st.title("AI Driver Safety Monitor 🚗💤")
st.write("Mechanical Engineering Project — IIT Kharagpur")

st.sidebar.title("Settings")
# Adding a manual play button to test if sound file is even reachable
test_sound = st.sidebar.button("🔊 Test Sound Manually")
if test_sound:
    st.markdown(get_audio_html("Amelia Island.mp3"), unsafe_allow_html=True)

ctx = webrtc_streamer(key="drowsiness-det", video_transformer_factory=DrowsinessTransformer)

# 2. Main Alert Check
if ctx.video_transformer and ctx.video_transformer.counter >= 20:
    st.error("⚠️ WAKE UP! Drowsiness Detected.")
    # Forced Sound Injection
    st.markdown(get_audio_html("Amelia Island.mp3"), unsafe_allow_html=True)