import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
import av
import cv2
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Live Tracker", layout="wide")

@st.cache_resource
def load_model():
    # 'n' is the Nano version—crucial for real-time mobile performance
    return YOLO("yolov8n.pt")

model = load_model()

# --- SIDEBAR UI ---
st.sidebar.title("🎮 Controls")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.05)
# Allows cellphone users to flip between front (user) and back (environment) cameras
camera_facing = st.sidebar.selectbox("Select Camera", ["environment", "user"], index=0)

# --- VIDEO PROCESSING ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # 1. Performance Hack: Resize for Inference
    # Processing 1080p on a server CPU is slow. 640px is YOLO's native size.
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (640, int(640 * h / w)))

    # 2. Tracking with Persistence
    # persist=True allows the model to remember IDs across frames
    results = model.track(
        img_resized, 
        persist=True, 
        conf=conf_threshold, 
        verbose=False
    )

    # 3. Annotation
    # We plot on the resized image for speed
    annotated_frame = results[0].plot()
    
    # 4. (Optional) Convert back to original size if display looks blurry
    # annotated_frame = cv2.resize(annotated_frame, (w, h))

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- WEBRTC CONFIG ---
# STUN servers are mandatory for mobile data (4G/5G) to work
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.title("📱 Live Object Detection & Tracing")
st.markdown("Accessible on mobile browsers via HTTPS.")

# The 'key' includes camera_facing so it resets when you switch cameras
webrtc_streamer(
    key=f"yolo-tracker-{camera_facing}",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    async_processing=True,
    media_stream_constraints={
        "video": {
            "facingMode": camera_facing,
            "width": {"ideal": 640},
            "frameRate": {"ideal": 30}
        },
        "audio": False,
    },
)
