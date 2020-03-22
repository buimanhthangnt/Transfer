import os
import glob
import imageio
from keras import applications
# from face_detection import detect_face
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Flatten, Dense, Activation, Dropout, Input, BatchNormalization
from keras.models import Sequential, Model
import cv2
import pickle
import numpy as np
from keras import metrics
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
from model import tiny_XCEPTION, little_XCEPTION
# from my_generator2 import My_Generator

# net = applications.mobilenet_v2.MobileNetV2(include_top=False, pooling='avg', weights='imagenet',
#                                 input_shape = (223,223,3))
# net = applications.nasnet.NASNetMobile(input_shape=(223, 223, 3), include_top=False, weights='imagenet', 
#                                 pooling='avg')

# print(len(net.layers))
# model = Sequential()
# model.add(net)
# model.add(Dense(2, activation='softmax'))

# for layer in net.layers[:-45]:
#    layer.trainable = False
model = little_XCEPTION((112, 112, 3), 2)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
print("Compile model done!")


earlyStopping = EarlyStopping(monitor='val_acc', patience=20, verbose=1)
filepath = "models/gb_model.h5"
mcp_save = ModelCheckpoint(filepath, save_best_only=True, monitor='val_acc')
reduce_lr = ReduceLROnPlateau('val_acc', factor=0.5,
                                patience=4, verbose=1)

train_data_dir = 'data/train'
validation_data_dir = 'data/test'
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.1,
    #brightness_range=(0.7, 1.25),
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    shear_range=0.11,
    rotation_range=15,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(112, 112),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(112, 112),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle = False)

class_weights = class_weight.compute_class_weight(
                            'balanced',
                            np.unique(train_generator.classes), 
                            train_generator.classes)
print(class_weights)
print(train_generator.class_indices)

model.fit_generator(generator=train_generator, epochs=50, verbose=1, class_weight=class_weights, \
        callbacks=[earlyStopping, mcp_save, reduce_lr], validation_data=validation_generator)
