import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import cv2

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="object oriented detection", layout="wide", page_icon="🛡️")

# Custom CSS for a "Command Center" look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #3b82f6; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # Using the nano model for speed in web environments
    return YOLO("yolov8n.pt")

model = load_model()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Control Panel")
target_object = st.sidebar.selectbox("🚨 Alert Trigger Object", ["person", "cell phone", "keyboard", "suitcases", "laptop"])
conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5, 0.05)

# --- NEW FEATURE: MIRROR TOGGLE ---
mirror_view = st.sidebar.checkbox("🪞 Mirror View (Invert)", value=True)

enable_blur = st.sidebar.checkbox("Privacy Mode (Blur Background)")

# --- VIDEO CALLBACK LOGIC ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # --- IMPLEMENT MIRROR/INVERT ---
    if mirror_view:
        img = cv2.flip(img, 1) # 1 indicates horizontal flip
    
    # Run Inference on the (possibly flipped) image
    results = model.track(img, persist=True, conf=conf_threshold, verbose=False)
    
    # 1. OBJECT DETECTION LOGIC
    current_objects = results[0].boxes.cls.tolist()
    class_names = results[0].names
    alert_active = any(class_names[int(cls)] == target_object for cls in current_objects)
    
    # 2. ANNOTATION
    annotated_frame = results[0].plot()
    
    # Visual Alert Overlay
    if alert_active:
        cv2.rectangle(annotated_frame, (0,0), (img.shape[1], img.shape[0]), (0,0,255), 10)
        cv2.putText(annotated_frame, f"ALERT: {target_object.upper()} DETECTED", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- MAIN INTERFACE ---
st.title("🛡️ Object Oriented Detection")
st.divider()

main_col, side_col = st.columns([2, 1])

with main_col:
    ctx = webrtc_streamer(
        key="sentry-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with side_col:
    st.subheader("📋 System Status")
    if ctx.state.playing:
        st.success(f"Monitoring for: {target_object}")
        if mirror_view:
            st.info("🪞 Mirror Mode: ON")
    else:
        st.error("System Status: Offline")
