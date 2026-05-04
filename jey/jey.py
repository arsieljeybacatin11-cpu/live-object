import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
import av
import cv2
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Live Tracker", layout="wide", page_icon="📱")

@st.cache_resource
def load_model():
    # 'n' is the Nano version—crucial for real-time mobile/CPU performance
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

    # 1. PERFORMANCE DOWN-SAMPLING
    # Mobile browsers often send 1080p+, which chokes server CPUs.
    # We resize to YOLO's native 640px width to keep the FPS high.
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (640, int(640 * h / w)))

    # 2. TRACKING (With persistence for object IDs)
    results = model.track(
        img_resized, 
        persist=True, 
        conf=conf_threshold, 
        verbose=False
    )

    # 3. ANNOTATION
    # Plotting on the smaller image is faster
    annotated_frame = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- WEBRTC CONFIG ---
# STUN servers are MANDATORY for mobile data (4G/5G) or office networks to connect
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.title("📱 Live Object Detection & Tracing")
st.markdown("""
    This app supports **Mobile Devices** and **Desktop**. 
    * Adjust confidence in the sidebar.
    * Switch between front and back cameras.
""")

# The 'key' changes with camera_facing to force a clean reload when switching
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
