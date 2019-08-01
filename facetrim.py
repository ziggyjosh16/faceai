import os  
from PIL import Image  
import cv2 as cv   



def getfaces(imgdir, face_cascade=cv.CascadeClassifier('C:/Users/Joshua.868/AppData/Local/Continuum/anaconda3/pkgs/opencv-4.0.1-py37hb76ac4c_200/Library/etc/haarcascades/haarcascade_frontalface_default.xml')):
    for entry in imgdir:
        if entry.is_dir():
            num = 1
            testdir = entry.name+'_face_test'
            traindir = entry.name+'_face_train'
            try:
                os.mkdir(testdir)
                os.mkdir(traindir)
            except FileExistsError:
                print('Directory Already Exists.')
            for photo in os.scandir(entry.path):
                try:
                    img = Image.open(photo.path)  
                    matrix = cv.imread(photo.path)  
                    faces = face_cascade.detectMultiScale(
                        matrix, 1.3, 5)  
                    if(len(faces) == 1):  
                        x, y, w, h = faces[0]  
                        crpim = img.crop(
                            (x, y, x+w, y+h)).resize((64, 64))  
                        if(num % 10 == 0):
                            crpim.save(testdir + '/face' + str(num) + '.jpg')
                            num = num + 1
                        else:
                            crpim.save(traindir + '/face' + str(num) + '.jpg')
                            num = num+1
                except Exception as e:
                    print(e)


path = os.scandir(
    'C:/Users/Joshua.868/Documents/Projects/MachineLearning/WebScraper')
getfaces(path)
