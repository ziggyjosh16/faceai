
from matplotlib import pyplot as plt 
import numpy as np
import cv2 as cv

from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.preprocessing import image
from PIL import Image


model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu')) 
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 2, activation = 'softmax'))
model.compile(optimizer = 'adam', 
                        loss ='categorical_crossentropy', 
                     metrics = ['accuracy'])
model.summary()

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,  #魚度  
                                   zoom_range = 0.2,   #縮放   
                                   horizontal_flip = True 
                                  )
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(
    '../OpenCV/training/', target_size = (64, 64),
     batch_size = 7,
     class_mode = 'categorical')
test_set = test_datagen.flow_from_directory(
    '../OpenCV/testing/', target_size = (64, 64),
    batch_size = 7, 
    class_mode = 'categorical')


transform_dic = {
    'beyonce_face'  : 'Beyonce',
    'rihanna_face'    : 'Rihanna',
}
name_dic = {v:transform_dic.get(k) for k,v in training_set.class_indices.items()}
name_dic


early_stopping=EarlyStopping(monitor='val_loss',patience=50,verbose=1)  #終斷點, 50次沒有比上次好就停 v
history = model.fit_generator(training_set,
                         epochs=1000,
                         validation_steps=10,     #驗正 １０次　　依據你上面的flow_from_directory的設定　BATCH_SIZE=7,
                         samples_per_epoch = 30,  #訓練
                         verbose = 1,
                         validation_data = test_set,
                         callbacks=[early_stopping])
#validation_steps >>驗正 １０次　　依據你上面的flow_from_directory的設定　BATCH_SIZE=7, 隨機拿７張圖　做分類的驗正
#所以一個壘代　就會用到　１０＊７張圖
#samples_per_epoch　＞＞訓練用　同上做法　３０＊７張圖
#　以上拿在Flow_from_directory的圖　都隨機經過角度,縮放或旋轉產不一樣的圖片

face_cascade = cv.CascadeClassifier('C:/Users/Joshua.868/AppData/Local/Continuum/anaconda3/pkgs/opencv-4.0.1-py37hb76ac4c_200/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
filepath='ALLALL.JPG'
im = Image.open(filepath) #開圖 等一下要記錄 臉的位置
img = cv.imread(filepath) #轉矩陣
faces = face_cascade.detectMultiScale(img, 1.2, 3)
font = cv.FONT_HERSHEY_PLAIN
for x,y,w,h in faces:    # 像素x,y, w 寬  h 高
    box = (x, y, x+w, y+h)
    crpim = im.crop(box).resize((64,64))
    target_image = image.img_to_array(crpim)
    target_image = np.expand_dims(target_image, axis = 0) #分類器是四維,
    res = model.predict_classes(target_image)[0]
    cv.rectangle(img,(x,y),(x+w,y+h),(14,201,255),2)  #畫框 要用矩陣data
    cv.putText(img,name_dic.get(res), (x + int(w/3)-70, y-10), font, 1.5, (14,201,255), 3) # 1.5粗度,3大小
    #cv.putText(img,name_dic.get(res), (100, 110), font, 1.5, (14,201,255), 3) # 1.5粗度,3大小
# get_ipython().run_line_magic('pylab', 'inline')
plt.figure(figsize=(30,20)) #圖片大小
#plt.imshow(img)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)) #cv色為 bgr 轉rgb 色彩好一點
plt.show()


