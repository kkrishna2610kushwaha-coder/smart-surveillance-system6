import os
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Smart Surveillance System", page_icon="📹", layout="wide")
st.title("📹 Smart Surveillance System")
st.subheader("Employee Monitoring and Indexing Using Deep Learning")

DATABASE = "employee_records.csv"
COLUMNS = ["Date", "Time", "Person_Count", "Tracking_IDs", "Source"]


def initialize_database():
    if not os.path.exists(DATABASE):
        pd.DataFrame(columns=COLUMNS).to_csv(DATABASE, index=False)


def save_record(count, ids, source):
    initialize_database()
    row = {
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Person_Count": int(count),
        "Tracking_IDs": ", ".join(map(str, ids)),
        "Source": source,
    }
    pd.DataFrame([row]).to_csv(DATABASE, mode="a", header=False, index=False)


@st.cache_resource
def load_model():
    # Put yolo11n.pt beside app.py, or let Ultralytics download it automatically.
    return YOLO("yolo11n.pt")


with st.sidebar:
    st.header("Detection Settings")
    confidence = st.slider("Confidence threshold", 0.10, 0.90, 0.40, 0.05)
    st.caption("YOLO11 detects the person class only.")

try:
    model = load_model()
    st.sidebar.success("YOLO11n model loaded")
except Exception as error:
    st.error("Could not load YOLO11n. Check that yolo11n.pt is in the app folder or that the app can access the internet.")
    st.exception(error)
    st.stop()

initialize_database()
page = st.selectbox("Choose Module", ["Home", "Image Detection", "Video Monitoring", "Employee Records"])

if page == "Home":
    st.header("Home Dashboard")
    records = pd.read_csv(DATABASE)
    c1, c2, c3 = st.columns(3)
    c1.metric("AI Model", "YOLO11n")
    c2.metric("Monitoring Records", len(records))
    c3.metric("People Detected", int(records["Person_Count"].sum()) if not records.empty else 0)
    st.write("Upload an image or CCTV video to detect and track people.")
    st.info("This is a person-detection demo. It does not identify real-world employee identities or verify attendance.")

elif page == "Image Detection":
    st.header("Image Detection")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        if st.button("Detect People", type="primary"):
            results = model.predict(np.array(image), classes=[0], conf=confidence, verbose=False)
            result = results[0]
            count = len(result.boxes) if result.boxes is not None else 0
            annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
            st.image(annotated, caption="Detection result", use_container_width=True)
            st.success(f"People detected: {count}")
            save_record(count, [], uploaded.name)

elif page == "Video Monitoring":
    st.header("Video Monitoring and Tracking")
    uploaded = st.file_uploader("Upload a CCTV video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        st.video(uploaded)
        if st.button("Start Monitoring", type="primary"):
            temp_path = None
            cap = None
            try:
                suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                    temp.write(uploaded.getbuffer())
                    temp_path = temp.name

                cap = cv2.VideoCapture(temp_path)
                if not cap.isOpened():
                    st.error("Unable to open this video. Please try an MP4 file.")
                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    frame_view = st.empty()
                    progress = st.progress(0)
                    status = st.empty()
                    frame_no = 0
                    unique_ids = set()
                    max_people = 0
                    last_saved_second = -1
                    saved_records = 0

                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frame_no += 1
                        result = model.track(
                            frame, persist=True, classes=[0], conf=confidence,
                            tracker="bytetrack.yaml", verbose=False
                        )[0]
                        boxes = result.boxes
                        count = len(boxes) if boxes is not None else 0
                        max_people = max(max_people, count)
                        ids = []
                        if boxes is not None and boxes.id is not None:
                            ids = boxes.id.int().cpu().tolist()
                            unique_ids.update(ids)

                        annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
                        frame_view.image(annotated, channels="RGB", use_container_width=True)

                        second = int(frame_no / fps)
                        if second > last_saved_second:
                            save_record(count, ids, uploaded.name)
                            saved_records += 1
                            last_saved_second = second

                        if total_frames > 0:
                            progress.progress(min(frame_no / total_frames, 1.0))
                        status.write(f"Processed frame {frame_no} / {total_frames or 'unknown'}")

                    cap.release()
                    cap = None
                    progress.progress(1.0)
                    st.success("Video monitoring completed.")
                    a, b, c = st.columns(3)
                    a.metric("Unique tracking IDs", len(unique_ids))
                    b.metric("Maximum people in one frame", max_people)
                    c.metric("Records saved", saved_records)
                    st.caption("Tracking IDs are temporary video-track IDs, not real-world identities.")
            except Exception as error:
                st.error("Video processing failed.")
                st.exception(error)
            finally:
                if cap is not None:
                    cap.release()
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

elif page == "Employee Records":
    st.header("Employee Monitoring Records")
    records = pd.read_csv(DATABASE)
    if records.empty:
        st.info("No records yet. Run image or video detection first.")
    else:
        st.dataframe(records, use_container_width=True)
        st.download_button(
            "Download Surveillance Report CSV",
            records.to_csv(index=False).encode("utf-8"),
            file_name="surveillance_report.csv",
            mime="text/csv"
        )
        if st.button("Clear All Records"):
            pd.DataFrame(columns=COLUMNS).to_csv(DATABASE, index=False)
            st.success("Records cleared. Refresh the page.")

st.divider()
st.caption("Python • YOLO11 • OpenCV • Streamlit • Deep Learning")
