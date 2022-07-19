import streamlit as st
from main import * 

st.sidebar.header("Upload Image")

st.title("Face Detection")
st.markdown(
    """
    1) Detect faces in an image
    2) Draw facial landmarks
    """
)

image_upload = st.sidebar.file_uploader("Load your image here")

if image_upload is not None:
    image = Image.open(image_upload)
else:
    image = Image.open("./resources/example.jpg")
    
col1, col2, col3, col4 = st.columns([1,1,1,1])

with col1:
    draw_bb_button = st.button('Detect Faces')
with col2:
    draw_landmark_button = st.button('Draw Landmarks')

if draw_bb_button:
    with st.spinner("Detecting faces..."):
        result = draw_face_bb(image)
    if not result:
        with st.spinner("Failed to find faces, trying harder..."):
            result = draw_face_bb(image, number_of_times_to_upsample=3)
    if not result:
        st.header("No faces found")
        st.image(image)
    else:
        st.header("Image with detected faces")
        st.image(result)
    
elif draw_landmark_button:
    with st.spinner("Detecting landmarks..."):
        result = draw_face_landmarks(image, landmark_thickness=2)
    st.header("Image with facial landmarks")
    st.image(result)
else:
    st.header("Original Image")
    st.image(image)