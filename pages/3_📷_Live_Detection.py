import av
import cv2
import streamlit as st
from main import get_face_locations_fast
from streamlit_webrtc import webrtc_streamer

st.title("Webcam Live Face Detection")

def find_faces(frame):
    img = frame.to_ndarray(format="bgr24")
    faces = get_face_locations_fast(img)

    for (x, y, w, h) in faces:
        cv2.rectangle(img=img, pt1=(x, y), pt2=(
            x + w, y + h), color=(255, 0, 0), thickness=2)
            
    return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="object-detection",
    media_stream_constraints={"video": True, "audio": False},
    video_frame_callback=find_faces,
    async_processing=True
)
