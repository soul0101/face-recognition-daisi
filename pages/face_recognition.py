import streamlit as st
# from main import * 

# st.sidebar.header("Upload Image")
# st.title("Face Recognition")

# col1, col2= st.columns([1,1])

# with col1:
#     st.markdown("## Reference")
#     reference_face_upload = st.file_uploader("Upload an image with a face for reference")
#     if reference_face_upload is not None:
#         reference_face_image = Image.open(reference_face_upload)
#     else:
#         reference_face_image = Image.open("./resources/sample_reference_face.jpg")
#     st.image(reference_face_image)

# with col2:
#     st.markdown("## Authenticate")
#     check_face_upload = st.file_uploader("Upload image with face to authenticate")
#     if check_face_upload is not None:
#         check_face_image = Image.open(check_face_upload)
#     else:
#         check_face_image = Image.open("./resources/sample_check_face.jpg")
#     st.image(check_face_image)

# reference_encodings = get_face_encodings(reference_face_image)
# check_encodings = get_face_encodings(check_face_image)

# if len(reference_encodings) == 0:
#     st.write("No faces found!")
# elif len(reference_encodings) > 1:
#     st.write("Please upload a reference image with only one face!")
# elif len(check_encodings) == 0:
#     st.write("No faces found!")
# else:
#     authenticate_btn = st.button("Authenticate")
#     if authenticate_btn:
#         with st.spinner("Authenticating faces..."):
#             result = compare_faces(check_encodings, reference_encodings[0])
#         if True in result:
#             st.write("Authentication Succesful!")
#         else:
#             st.write("Authentication Failed!")

st.write('hi2')