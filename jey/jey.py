import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import cv2
import numpy as np
from collections import deque

# --- CONFIGURATION ---
st.set_page_config(page_title="YOLOv8 Live Tracker", layout="wide")

# Cache model to avoid reloading
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Sidebar for live adjustments
st.sidebar.title("🚀 Detection Settings")
conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5, 0.05)
show_traces = st.sidebar.checkbox("Show Motion Traces", value=True)

# Store tracking history for the "Tracing" effect
if "track_history" not in st.session_state:
    st.session_state.track_history = {}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # 1. OPTIMIZATION: Resize frame for faster processing
    # Processing at a lower resolution significantly boosts FPS
    input_img = cv2.resize(img, (640, 480))

    # 2. INFERENCE
    results = model.track(
        input_img,
        persist=True,
        conf=conf_threshold,
        verbose=False
    )

    # 3. CUSTOM ANNOTATION & TRACING
    annotated_frame = results[0].plot()

    # Logic for Drawing Traces (Lines following the object)
    if show_traces and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            
            # Update history for this specific ID
            if track_id not in st.session_state.track_history:
                st.session_state.track_history[track_id] = deque(maxlen=20)
            st.session_state.track_history[track_id].append((float(x), float(y)))

            # Draw the trace line
            points = np.array(st.session_state.track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- UI LAYOUT ---
st.title("🎥 Live Object Detection & Tracing")
st.markdown("""
This app uses **YOLOv8** to track objects and **WebRTC** to provide a low-latency stream.
Adjust the confidence slider in the sidebar to filter detections.
""")

webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)
