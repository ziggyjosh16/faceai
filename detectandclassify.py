import cv2 as cv
import json
from matplotlib import pylab as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image


with open('names.json', 'r') as f:
    name_dic = json.load(f)

print(name_dic)
model = load_model('mymodel.hd5')
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
filepath='ALLALL.jpg'
im = Image.open(filepath) 
img = cv.imread(filepath) 
faces = face_cascade.detectMultiScale(img, 1.2, 3)
font = cv.FONT_HERSHEY_PLAIN
for x,y,w,h in faces:    
    box = (x, y, x+w, y+h)
    crpim = im.crop(box).resize((64,64))
    target_image = image.img_to_array(crpim)
    target_image = np.expand_dims(target_image, axis = 0) 
    res = model.predict_classes(target_image)[0]
    print(res)
    cv.rectangle(img,(x,y),(x+w,y+h),(14,201,255),2)  
    cv.putText(img,name_dic.get(str(res)), (x + int(w/3)-70, y-10), font, 1.5, (14,201,255), 3) 
plt.figure(figsize=(30,20)) 
#plt.imshow(img)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)) 
plt.show()