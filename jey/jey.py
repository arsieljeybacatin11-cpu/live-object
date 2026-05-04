import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
import av
import cv2

# --- PAGE CONFIG ---
st.set_page_config(page_title="Mobile AI Tracker", layout="centered")

@st.cache_resource
def load_model():
    # Nano model is essential for mobile/server CPU performance
    return YOLO("yolov8n.pt")

model = load_model()

# --- MOBILE UI CONTROLS ---
st.title("📱 Mobile Live Detection")

# Sidebar for mobile-friendly adjustments
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.4, 0.05)
# Use the "environment" facing mode for the back camera by default
facing_mode = st.sidebar.selectbox("Camera", ["environment", "user"], index=0)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # 1. SCALE FOR MOBILE PERFORMANCE
    # Mobile browsers often send high-res video that chokes the server CPU.
    # Downscaling to 480px keeps the frame rate high.
    h, w = img.shape[:2]
    scale = 480 / max(h, w)
    if scale < 1:
        img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        img_resized = img

    # 2. INFERENCE
    results = model.track(img_resized, persist=True, conf=confidence, verbose=False)

    # 3. ANNOTATE
    annotated_frame = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- WEBRTC CONFIGURATION ---
# STUN servers are mandatory for mobile networks (4G/5G) to bypass NAT
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="mobile-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    async_processing=True, # Critical for mobile to prevent video lag
    media_stream_constraints={
        "video": {
            "facingMode": facing_mode, # "environment" = Back Camera, "user" = Front
            "width": {"ideal": 640},
            "frameRate": {"ideal": 20}
        },
        "audio": False
    },
)

st.info("💡 Hint: If the video is slow, lower the confidence or ensure you are on Wi-Fi.")
