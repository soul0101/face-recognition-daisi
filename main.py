import cv2
import numpy as np
from PIL import Image
import streamlit as st
import face_recognition

def process_image(image: Image.Image or np.ndarray) -> np.ndarray:
    """
    Internal function to process a passed image

    Parameters:
        image: the image to be processed 
        
    Returns:
        np.ndarray
    """

    if image is None:
        raise Exception("Image object is None.\n Provide valid image object.")
    if type(image) == np.ndarray:
        return image

    return np.asarray(image)


def get_face_locations(image):
    """
    Returns an array of bounding boxes of human faces in a image

    Parameters:
        image: the image object to be processed 
        
    Returns:
        A list of tuples of found face locations in css (top, right, bottom, left) order
    """

    image = process_image(image)
    return face_recognition.face_locations(image)

def get_face_encodings(image):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    Parameters:
        image: the image object to be processed 
        
    Returns:
       A list of 128-dimensional face encodings (one for each face in the image)
    """

    image = process_image(image)
    return face_recognition.face_encodings(image)

def get_face_landmarks(image):
    """
    Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in the image

    Parameters:
        image: the image object to be processed 
        
    Returns:
       A list of dicts of face feature locations (eyes, nose, etc)
    """

    image = process_image(image)
    return face_recognition.face_landmarks(image)

def compare_faces(face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    Parameters:
        face_encodings: A list of known face encodings
        face_encoding_to_check: A single face encoding to compare against the list
        tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
        
    Returns:
        A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return face_recognition.compare_faces(face_encodings, face_encoding_to_check, tolerance)

def similiarity_faces(face_encodings, face_encoding_to_check):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance for each comparison face. 
    The distance tells you how similar the faces are.

    Parameters:
        face_encodings: A list of known face encodings
        face_encoding_to_check: A single face encoding to compare against the list
        
    Returns:
        A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    return face_recognition.face_distance(face_encodings, face_encoding_to_check)

def draw_face_bb(image, box_color=(0, 255, 0), box_thickness=3, return_type="pil"):
    """
    Given an image, draws bounding boxes on all detected faces

    Parameters:
        image: the image object to be processed 
        box_color (optional): tuple containing the rgb values for box color
        box_thickness (optional): thickness of bounding box 
        return_type (optional): "pil" - Return image to be of the type PIL.Image.Image
                     "np.ndarray" - Return image to be of the type np.ndarray
        
    Returns:
       The image object with bounding boxes drawn
    """

    image = process_image(image)
    face_locations = get_face_locations(image)
    for location in face_locations:
        cv2.rectangle(image, (location[3], location[0]), (location[1], location[2]), box_color, box_thickness)

    if return_type == "pil":
        return Image.fromarray(image)
    else:
        return image

def draw_face_landmarks(image, landmark_color=(0, 255, 0), landmark_thickness=-1, return_type="pil"):
    """
    Given an image, draws facial landmarks

    Parameters:
        image: the image object to be processed 
        landmark_color (optional): tuple containing the rgb values for box color
        landmark_thickness (optional): thickness of bounding box 
        return_type (optional): "pil" - Return image to be of the type PIL.Image.Image
                     "np.ndarray" - Return image to be of the type np.ndarray
        
    Returns:
       The image object with facial landmarks
    """

    image = process_image(image)
    face_landmarks = get_face_landmarks(image)
    for face in face_landmarks:
        for feature, landmarks in face.items():
            for landmark in landmarks:
                cv2.circle(image, landmark, 1, landmark_color, landmark_thickness)

    if return_type == "pil":
        return Image.fromarray(image)
    else:
        return image

# img = Image.open("./face-recognition-daisi/group2.jpg")
# img2 = np.asarray(img)

# final = draw_face_landmarks(img)
# final.show()

################################## UI ##############################################

if __name__ == "__main__":
    st.title("Face Recognition")
    st.write("This Daisi allows you to provide an image, and detect and recognize faces, extract facial landmarks, draw bounding boxes and much more! Upload an image with a face to get started!")

    image_upload = st.sidebar.file_uploader("Load your image here")
    if image_upload is not None:
        image = Image.open(image_upload)
        st.header("Image")
        st.image(image)
        with st.spinner("Colorizing your image"):
            result = draw_face_bb(image)
        st.header("Image with bounding boxes")
        st.image(result, caption='Faces detected')


