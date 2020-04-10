
from matplotlib import pyplot as plt 
import numpy as np
import cv2 as cv
import json
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
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
                                   shear_range = 0.2,    
                                   zoom_range = 0.2,   
                                   horizontal_flip = True 
                                  )
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(
    'training/', target_size = (64, 64),
     batch_size = 7,
     class_mode = 'categorical')
test_set = test_datagen.flow_from_directory(
    'testing/', target_size = (64, 64),
    batch_size = 7, 
    class_mode = 'categorical')


transform_dic = {
    'beyonce_face_train'  : 'Beyonce',
    'rihanna_face_train'    : 'Rihanna',
}
name_dic = {v:transform_dic.get(k) for k,v in training_set.class_indices.items()}

with open('names.json', 'w') as f:
        json.dump(name_dic, f)

early_stopping=EarlyStopping(monitor='val_loss',patience=50,verbose=1)  
history = model.fit_generator(training_set,
                         epochs=100,
                         validation_steps=10,     
                         steps_per_epoch = 30,  
                         verbose = 1,
                         validation_data = test_set,
                         callbacks=[early_stopping])

model.save('mymodel.hd5')


