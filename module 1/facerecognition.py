import cv2
import numpy as np
import face_recognition
import os

class faceRec:
    def encodeImages(self,path):
        images = []
        classnames = []
        mylist = os.listdir(path)
        # print(mylist)

        for cl in mylist:
            curimg = cv2.imread(f'{path}/{cl}')
            images.append(curimg)
            classnames.append(os.path.splitext(cl)[0])
        # print(classnames)

        encodelist = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodelist.append(encode)
        # return encodelist,classnames
        np.save("./trained/encodelist.npy", encodelist)
        np.save("./trained/classnames.npy", classnames)

    def detectAndRecognize(self,encodelistpath,classnamepath):
        encodlistknown = np.load(encodelistpath)
        classnames = np.load(classnamepath)
        names = []
        cap = cv2.VideoCapture(0)
        while True:
            success,img = cap.read()
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
                    y1, x2, y2, x1 = faceloc
                    y1, x2, y2, x1 = y1, x2, y2, x1
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                    names.append(name)
            return names
            cv2.imshow("Webcam", img)
            cv2.waitKey(1)
