import cv2
import numpy as np
import face_recognition
import os
classnames = np.load("classnames.npy")
encodlistknown = np.load("encoding.npy")
names = []

img = cv2.imread('training/group img 1.jpeg')
imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
facesCurrFrame = face_recognition.face_locations(imgS)
encodesCurrFrame = face_recognition.face_encodings(imgS,facesCurrFrame)

for encodeFace, faceloc in zip(encodesCurrFrame, facesCurrFrame):
    matches = face_recognition.compare_faces(encodlistknown, encodeFace)
    faceDis = face_recognition.face_distance(encodlistknown, encodeFace)
    # print(faceDis)
    matchIndex = np.argmin(faceDis)

    if matches[matchIndex]:
        name = classnames[matchIndex]
        names.append(name)

        # print(name)
        y1, x2, y2, x1 = faceloc
        y1, x2, y2, x1 = y1, x2, y2, x1
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.rectangle(img, (x1, y1 - 35), (x2, y2), (0, 255, 0), cv2.)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
print(names)
cv2.imshow("Webcam",img)
cv2.waitKey(1)