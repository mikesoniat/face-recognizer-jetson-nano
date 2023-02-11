# AI Face Recognition in Live Video with Jetson Nano
# Using OpenCV and face_recognition to identify known faces in live video
# Mike Soniat
# 2022

import face_recognition
import cv2
import numpy as np
import os
import pickle
import time 

# read training data from pickled file
with open('train.pkl','rb') as f:
    known_face_names = pickle.load(f)
    known_face_encodings = pickle.load(f)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
font=cv2.FONT_HERSHEY_SIMPLEX
process_this_frame = True

# Get a reference to webcam
cam1 = cv2.VideoCapture(1)

# adjust frame size: 640x480, 800x600, 1280x720, 1920x1080
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# adjust scale factor: .25, .5, .75, 1
scaleFactor = .75

while True:
    # Grab a single frame of video
    ret, frame = cam1.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=scaleFactor, fy=scaleFactor)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown Person"

            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        # add back scale factor
        top = int(top/scaleFactor)
        right = int(right/scaleFactor)
        bottom = int(bottom/scaleFactor)
        left = int(left/scaleFactor)

        # draw rectangle and text on faces
        cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
        cv2.putText(frame, name, (left, top-6), font, .5, (255,0,0), 2)        

    # Display the resulting image
    cv2.imshow('Live Video', frame)
    cv2.moveWindow('Live Video', 0,0)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cam1.release()
cv2.destroyAllWindows()
