import os  # 建立文件，刪除文件，查詢文件"
from PIL import Image  # 從phthoy 資料找 叫出imgge相關功能
import cv2 as cv   # as新的名字
# CascadeClassifier 是funcation 裡面的 haarcascade_frontalface_default 就是人臉特真位置資料
# 人臉  有眼 鼻子 口  耳 相對位置 寫在xml


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
                    img = Image.open(photo.path)  # 先開圖片
                    matrix = cv.imread(photo.path)  # 再轉矩陣
                    faces = face_cascade.detectMultiScale(
                        matrix, 1.3, 5)  # 使用funcation
                    if(len(faces) == 1):  # 抓一個人頭圖片
                        x, y, w, h = faces[0]  # x,y w寛h高 特真值
                        crpim = img.crop(
                            (x, y, x+w, y+h)).resize((64, 64))  # 存成 64*64的圖片
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
