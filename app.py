import streamlit as st
import cv2
import time
from drowsiness_detect import process_frame

st.set_page_config(page_title="Driver Drowsiness Detection", layout="centered")

st.title("🚗 Driver Drowsiness Detection System")
st.write("Real-time Eye + Yawning Detection")

start = st.button("Start Camera")
stop = st.button("Stop Camera")

FRAME_WINDOW = st.empty()

if "run" not in st.session_state:
    st.session_state.run = False

if start:
    st.session_state.run = True

if stop:
    st.session_state.run = False

camera = cv2.VideoCapture(0)

while st.session_state.run:
    ret, frame = camera.read()
    if not ret:
        st.error("Camera not accessible")
        break

    frame = process_frame(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    FRAME_WINDOW.image(frame)

camera.release()
