from facerecognition import faceRec

# Run line number 5 only once while running the code for the first time, after running the code for the first time
# comment out the line number 5 and then run the code.

faceRec.encodeImages([],'./images')


names = faceRec.detectAndRecognize([],'./trained/encodelist.npy','./trained/classnames.npy')
print(set(names))
