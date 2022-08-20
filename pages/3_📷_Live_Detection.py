import av
import cv2
import streamlit as st
from main import get_face_locations_fast
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

st.title("Webcam Live Face Detection")

def find_faces(frame):
    img = frame.to_ndarray(format="bgr24")
    faces = get_face_locations_fast(img)

    for (x, y, w, h) in faces:
        cv2.rectangle(img=img, pt1=(x, y), pt2=(
            x + w, y + h), color=(255, 0, 0), thickness=2)
            
    return av.VideoFrame.from_ndarray(img, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}, 
                    {"urls": ["stun:stun3.l.google.com:19302"]},
                    {"urls": ["stun:stun.voipzoom.com:3478"]},
                    {"urls": ["stun:stun.halonet.pl:3478"]},
                    {"urls": ["stun:stun.altar.com.pl:347"]}
                    ]
    }
)

ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_frame_callback=find_faces,
    async_processing=True
)
