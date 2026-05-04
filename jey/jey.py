import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
import av
import cv2
import numpy as np
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Vision Hub", layout="wide")

@st.cache_resource
def load_model():
    # 'n' is Nano (fastest), 's' is Small (better accuracy)
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🚀 AI Live & Image Detection")

# --- SHARED SETTINGS ---
st.sidebar.header("Configuration")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.05)

# --- TABS INTERFACE ---
tab1, tab2 = st.tabs(["🎥 Live Stream", "🖼️ Image Upload"])

# --- TAB 1: LIVE STREAM ---
with tab1:
    st.subheader("Real-time Camera Detection")
    
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")

        # Performance Hack: Resize for Inference
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (640, int(640 * h / w)))

        # YOLO Tracking
        results = model.track(img_resized, persist=True, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="live-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        async_processing=True,
        media_stream_constraints={"video": True, "audio": False},
    )

# --- TAB 2: IMAGE UPLOAD ---
with tab2:
    st.subheader("Static Image Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert uploaded file to OpenCV format
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # YOLO only accepts BGR, PIL provides RGB
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Run Detection
        with st.spinner("Analyzing image..."):
            results = model.predict(img_bgr, conf=conf_threshold)
            
            # Draw results
            res_plotted = results[0].plot()
            # Convert back to RGB for Streamlit display
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        # Display result
        st.image(res_rgb, caption="Detected Results", use_container_width=True)
        
        # Metadata display
        if len(results[0].boxes) > 0:
            st.success(f"Detected {len(results[0].boxes)} objects!")
        else:
            st.warning("No objects detected at this confidence level.")
