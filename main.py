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


def get_face_locations(image, number_of_times_to_upsample=1, model='hog'):
    """
    Returns an array of bounding boxes of human faces in a image

    Parameters:
        image: the image object to be processed 
        number_of_times_to_upsample (optional): How many times to upsample the image 
                looking for faces. Higher numbers find smaller faces.
        model (optional): Which face detection model to use. “hog” is less accurate 
                but faster on CPUs. “cnn” is a more accurate deep-learning model which is 
                GPU/CUDA accelerated (if available). The default is “hog”.
        
    Returns:
        A list of tuples of found face locations in css (top, right, bottom, left) order
    """

    image = process_image(image)
    return face_recognition.face_locations(image, number_of_times_to_upsample=number_of_times_to_upsample, model=model)

def get_face_encodings(image, known_face_locations=None, num_jitters=1, model='small'):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    Parameters:
        image: the image object to be processed 
        known_face_locations (optional): the bounding boxes of each face if you already know them.
        num_jitters (optional): How many times to re-sample the face when calculating encoding. Higher 
                is more accurate, but slower (i.e. 100 is 100x slower)
        model (optional): which model to use. “large” or “small” (default) which only returns 5 
                points but is faster.
        
    Returns:
       A list of 128-dimensional face encodings (one for each face in the image)
    """

    image = process_image(image)
    return face_recognition.face_encodings(image, known_face_locations=known_face_locations, num_jitters=num_jitters, model=model)

def get_face_landmarks(image, face_locations=None, model='large'):
    """
    Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in the image

    Parameters:
        image: the image object to be processed 
        
    Returns:
       A list of dicts of face feature locations (eyes, nose, etc)
    """

    image = process_image(image)
    return face_recognition.face_landmarks(image, face_locations=face_locations, model=model)

def compare_faces(face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    Parameters:
        face_encodings: A list of known face encodings
        face_encoding_to_check: A single face encoding to compare against the list
        tolerance (optional): How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
        
    Returns:
        A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return face_recognition.compare_faces(face_encodings, face_encoding_to_check, tolerance=tolerance)

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

def draw_face_bb(image, number_of_times_to_upsample=1, model='hog', box_color=(0, 255, 0), box_thickness=3, return_type="pil"):
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
    face_locations = get_face_locations(image, number_of_times_to_upsample=number_of_times_to_upsample, model=model)
    
    if not face_locations:
        return None

    for location in face_locations:
        cv2.rectangle(image, (location[3], location[0]), (location[1], location[2]), box_color, box_thickness)

    if return_type == "pil":
        return Image.fromarray(image)
    else:
        return image

def draw_face_landmarks(image, face_locations=None, model='large', landmark_color=(0, 255, 0), landmark_thickness=1, return_type="pil"):
    """
    Given an image, draws facial landmarks

    Parameters:
        image: the image object to be processed 
        landmark_color (optional): tuple containing the rgb values for box color
        landmark_thickness (optional): thickness of bounding box 
        model (optional): which model to use. “large” (default) or “small” which only returns 5 points but is faster.
        return_type (optional): "pil" - Return image to be of the type PIL.Image.Image
                     "np.ndarray" - Return image to be of the type np.ndarray
        
    Returns:
       The image object with facial landmarks
    """

    image = process_image(image)
    face_landmarks = get_face_landmarks(image, face_locations=face_locations, model=model)
    for face in face_landmarks:
        for feature, landmarks in face.items():
            for landmark in landmarks:
                cv2.circle(image, landmark, landmark_thickness, landmark_color, -1)

    if return_type == "pil":
        return Image.fromarray(image)
    else:
        return image

################################## UI ##############################################

if __name__ == "__main__":
    import streamlit as st

    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to Face Recognition Daisi! 👋")

    st.markdown(
        """
            This daisi allows you to perform face-recognition operations on an image. 
            1) Face detection
            2) Facial landmark detection
            3) Face recognition
            4) Face encoding extraction
        """
    )