# Face Recognition Daisi
This Daisi brings all the essential face recognition features in a simple and reusable format. <br>
Since the python library 'face_recognition' depends on dlib which is written in C++, it can be tricky to deploy an app using it to a cloud hosting providers like Heroku or AWS.
The daisi solves the above problem and provides rich features such as: <br>

1) Face detection with bounding box locations
2) Facial landmark detection
3) Extract 128-dimensional face encodings from an image
4) Detect if given face encoding exists in the image (Facial recognition)
5) Face similarity index
6) Face Authentication and Recognition

## Test API Call
```python
import pydaisi as pyd
from PIL import Image

face_recognition = pyd.Daisi("soul0101/Face Recognition")

# Draw landmarks on detected faces
img = Image.open(r"path to image")
final = face_recognition.draw_face_landmarks(img).value
final.show()

# Draw bounding boxes on detected faces
img = Image.open(r"path to image")
final = face_recognition.draw_face_bb(img).value
final.show()

# Check if any face from img2 matches the face in img1
img1 = Image.open(r"path to image with reference face")
img2 = Image.open(r"path to image with face to check")
reference_encodings = face_recognition.get_face_encodings(img1).value
check_encodings = face_recognition.get_face_encodings(img2).value
result = face_recognition.compare_faces(check_encodings, reference_encodings[0]).value

if True in result:
    print("Authentication Succesful!")
else:
    print("Authentication Failed!")
```
