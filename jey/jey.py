import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
import av
import cv2

# --- PAGE CONFIG ---
st.set_page_config(page_title="YOLOv8 Live Live", layout="wide")

# Cache the model to prevent memory leaks on rerun
@st.cache_resource
def load_model():
    # 'yolov8n' is the fastest; 'yolov8s' is more accurate but slower
    return YOLO("yolov8n.pt")

model = load_model()

# --- SIDEBAR UI ---
st.sidebar.title("Configuration")
conf_val = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
# Filter specific classes (optional)
selected_classes = st.sidebar.multiselect(
    "Filter Objects", 
    list(model.names.values()), 
    default=["person", "cell phone", "laptop"]
)

# --- VIDEO CALLBACK ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # 1. OPTIMIZATION: Reduce image size for inference
    # Processing at 480p is significantly faster than 1080p
    orig_h, orig_w = img.shape[:2]
    input_img = cv2.resize(img, (640, 480))

    # 2. INFERENCE
    # Get class IDs for filtering
    class_ids = [k for k, v in model.names.items() if v in selected_classes]
    
    results = model.track(
        input_img,
        persist=True,
        conf=conf_val,
        classes=class_ids if selected_classes else None,
        verbose=False
    )

    # 3. ANNOTATION
    # The .plot() method returns the image with boxes
    annotated_img = results[0].plot()
    
    # Resize back to original view if needed (optional)
    if (orig_h, orig_w) != (480, 640):
        annotated_img = cv2.resize(annotated_img, (orig_w, orig_h))

    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# --- WEBRTC SETUP ---
# STUN servers help the browser and server find each other over the internet
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.title("🚀 Real-Time Object Access")
st.write("Using YOLOv8 for low-latency tracking.")

ctx = webrtc_streamer(
    key="live-tracking",
    mode=WebRtcMode.SENDRECV, # Handles both sending cam and receiving processed video
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.state.playing:
    st.success("Stream is active!")
