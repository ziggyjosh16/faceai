import os
from PIL import Image
import cv2 as cv


def is_illegal_dir(name):
    return ('_face_test'  in name
            or '_face_train' in name
            or '.git' in name
            or 'training' in name
            or 'testing' in name)


def getfaces(imgdir, face_cascade=cv.CascadeClassifier('haarcascade_frontalface_default.xml')):
    try:
        os.mkdir('testing')
        os.mkdir('training')
    except FileExistsError:
        print('Directories Exist')

    for entry in imgdir:
        if entry.is_dir() and is_illegal_dir(entry.name) == False:
            num = 1
            testdir = 'testing/' + entry.name + '_face_test'
            traindir = 'training/' + entry.name + '_face_train'
            try:
                os.mkdir(testdir)
                os.mkdir(traindir)
            except FileExistsError:
                print('Directory Already Exists.')
            for photo in os.scandir(entry.path):
                try:
                    if photo.path.lower().endswith('.jpg'):
                        img = Image.open(photo.path)
                        matrix = cv.imread(photo.path)
                        faces = face_cascade.detectMultiScale(
                            matrix, 1.3, 5)
                        if(len(faces) == 1):
                            x, y, w, h = faces[0]
                            crpim = img.crop(
                                (x, y, x+w, y+h)).resize((64, 64))
                            if(num % 10 == 0):
                                crpim.save(testdir + '/face' +
                                           str(num) + '.jpg')
                                num = num + 1
                            else:
                                crpim.save(traindir + '/face' +
                                           str(num) + '.jpg')
                                num = num+1
                except Exception as e:
                    print(e)


path = os.scandir('.')
getfaces(path)
