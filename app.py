import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import os

# 1. EAR Logic (Same as before)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# 2. Processor Class for Web Stream
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        # Ensure path is correct for deployment
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])
            
            # Simplified EAR check for demo
            leftEAR = eye_aspect_ratio(shape[42:48])
            rightEAR = eye_aspect_ratio(shape[36:42])
            ear = (leftEAR + rightEAR) / 2.0

            if ear < 0.25:
                cv2.putText(img, "DROWSY!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            cv2.putText(img, f"EAR: {ear:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        return img

st.title("AI Driver Safety Monitor 🚗")
st.write("Mechanical Engineering Project - IIT Kharagpur")

webrtc_streamer(key="example", video_transformer_factory=VideoProcessor)