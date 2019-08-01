
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


early_stopping=EarlyStopping(monitor='val_loss',patience=50,verbose=1)  
history = model.fit_generator(training_set,
                         epochs=1000,
                         validation_steps=10,     
                         samples_per_epoch = 30,  
                         verbose = 1,
                         validation_data = test_set,
                         callbacks=[early_stopping])


face_cascade = cv.CascadeClassifier('C:/Users/Joshua.868/AppData/Local/Continuum/anaconda3/pkgs/opencv-4.0.1-py37hb76ac4c_200/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
filepath='ALLALL.JPG'
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
    cv.rectangle(img,(x,y),(x+w,y+h),(14,201,255),2)  
    cv.putText(img,name_dic.get(res), (x + int(w/3)-70, y-10), font, 1.5, (14,201,255), 3) 
    
plt.figure(figsize=(30,20)) 
#plt.imshow(img)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)) 
plt.show()


