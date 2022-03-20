from facerecognition import faceRec
faceRec.encodeImages([],'./images')
names = faceRec.detectAndRecognize([],'./trained/encodelist.npy','./trained/classnames.npy')
print(names)
