import streamlit as st
import cv2
from ultralytics import YOLO

# Page configuration
st.set_page_config(page_title="AI Object Detector", layout="wide")
st.title("🚀 Real-Time Object Detection & Tracking")
st.write("This app uses YOLOv8 to detect objects in real-time from your webcam.")

# Sidebar for model settings
st.sidebar.header("Model Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Load the YOLOv8 model (Nano version is fastest for real-time)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Control buttons
run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

# Initialize Webcam
camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Failed to access webcam.")
        break

    # Convert frame to RGB (Streamlit and YOLO use RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform Object Detection and Tracking
    results = model.track(frame, persist=True, conf=confidence)

    # Plot results on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame in the Streamlit app
    FRAME_WINDOW.image(annotated_frame)
else:
    st.info("Webcam stopped. Check the box to start.")
    camera.release()
