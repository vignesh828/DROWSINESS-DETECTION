import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import os
import base64

# 1. Sound Logic for Browser (JavaScript + HTML5 Injection)
def get_audio_html(audio_file):
    if os.path.exists(audio_file):
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode()
            # JavaScript inclusion to force play and bypass some autoplay blocks
            return f'''
                <audio id="alert-audio" autoplay="true" style="display:none;">
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                </audio>
                <script>
                    var audio = document.getElementById("alert-audio");
                    audio.play().catch(error => {{
                        console.log("Autoplay blocked. Interaction required.");
                    }});
                </script>
            '''
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
        # GitHub root lo .dat file undali
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.counter = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])
            
            leftEAR = eye_aspect_ratio(shape[42:48])
            rightEAR = eye_aspect_ratio(shape[36:42])
            ear = (leftEAR + rightEAR) / 2.0

            # Draw visual guides
            leftHull = cv2.convexHull(shape[42:48])
            rightHull = cv2.convexHull(shape[36:42])
            cv2.drawContours(img, [leftHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightHull], -1, (0, 255, 0), 1)

            if ear < 0.25:
                self.counter += 1
                if self.counter >= 20:
                    cv2.putText(img, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.counter = 0

            cv2.putText(img, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return img

# --- Streamlit UI ---
st.set_page_config(page_title="AI Driver Safety Monitor", layout="centered")
st.title("AI Driver Safety Monitor 🚗💤")
st.write("Mechanical Engineering Project — IIT Kharagpur")

# 4. Mandatory User Interaction for Sound
st.sidebar.title("Configuration")
if st.sidebar.button("🔔 Enable Alert Sound"):
    st.sidebar.success("Sound Permissions Granted!")
    st.session_state['sound_enabled'] = True

# Start Video Streamer
ctx = webrtc_streamer(key="drowsiness-det", video_transformer_factory=DrowsinessTransformer)

# 5. Check if sound should be played
if ctx.video_transformer and ctx.video_transformer.counter >= 20:
    # IMPORTANT: Match your actual file name exactly
    audio_filename = "Amelia Island.mp3" 
    st.markdown(get_audio_html(audio_filename), unsafe_allow_html=True)
    st.error("⚠️ WAKE UP! Drowsiness Detected.")