# AI Face Recognition with Jetson Nano
# Using OpenCV and face_recognition to identify known faces
# Mike Soniat
# 2022

import face_recognition
import cv2
import os
import pickle 

Encodings=[]
Names=[]

# comment out training code after saving with Pickle
# train on known images
print("Training...")
known_dir = '/home/mikes/Desktop/PyPro/faceRecognizer/demoImages/known'
for root, dirs, files in os.walk(known_dir):
    for file in files:        
        path = os.path.join(root, file)
        name = os.path.splitext(file)[0]
        print(name)
        person = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(person)[0]
        
        # load arrays
        Encodings.append(encoding)
        Names.append(name)

# save training data using pickle
with open('train.pkl','wb') as f:
    pickle.dump(Names, f)
    pickle.dump(Encodings, f)

# read training data from pickled file
with open('train.pkl','rb') as f:
    Names = pickle.load(f)
    Encodings = pickle.load(f)

# set font for name labels
font=cv2.FONT_HERSHEY_SIMPLEX

# step through unknown files
unknown_dir = '/home/mikes/Desktop/PyPro/faceRecognizer/demoImages/unknown'
for root, dirs, files in os.walk(unknown_dir):
    for file in sorted(files):
        print(file)
        testImagePath = os.path.join(root, file)
        testImage = face_recognition.load_image_file(testImagePath)

        # find faces in unknown pic and encode
        face_positions = face_recognition.face_locations(testImage)
        allEncodings = face_recognition.face_encodings(testImage, face_positions)

        # convert pic to BGR (for OpenCV)
        testImage = cv2.cvtColor(testImage, cv2.COLOR_RGB2BGR)

        # step through faces and names in unknown pic
        for (top,right,bottom,left), face_encoding in zip(face_positions, allEncodings):
            name = 'Unknown Person'
            matches = face_recognition.compare_faces(Encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = Names[first_match_index]

            cv2.rectangle(testImage, (left,top),(right,bottom),(0,0,255), 2)
            cv2.putText(testImage, name, (left,top-6), font, 1, (0,255,255), 2)

        # show next image in window
        cv2.imshow(file, testImage)
        cv2.moveWindow(file,0,0)
        
        # wait for user to press a key
        cv2.waitKey(0)
        cv2.destroyAllWindows()     
