import os

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import Input, Model, optimizers
from keras.callbacks import EarlyStopping
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D, UpSampling2D, concatenate, core)
from keras.models import Sequential
from keras.optimizers import SGD

early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2040)])

dir_list = ["/home/liux/文档/项目/Classification-of-laryngeal-carcinoma/data/non_carcinoma/white_light/datagen/",
"/home/liux/文档/项目/Classification-of-laryngeal-carcinoma/data/carcinoma/white_light/datagen/"]

def get_img_path(dir_list):
    img_path_non = [dir_list[0]+i for i in os.listdir(dir_list[0])]
    img_paths = [dir_list[1]+i for i in os.listdir(dir_list[1])]
    img_paths.extend(img_path_non)
    img_paths = np.array(img_paths)
    index = np.arange(img_paths.shape[0])
    np.random.shuffle(index)
    img_paths = img_paths[index]
    return img_paths


def generate_arrays_from_file(img_paths_ll):
    while True:
        for img_path in img_paths_ll:
            img = np.random.randint(5,size=(6,28,28,1))
            label= np.random.randint(2,size=(6,2))

            yield (img,label)



def custom_net(img_height,img_width,img_ch):
    inputs = Input(shape=(img_height,img_width,img_ch))
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(8,(3,3),activation="relu",padding="same")(conv1)
    pool1 = MaxPooling2D(2)(conv1)

    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(2)(conv2)

    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(2)(conv4)

    flatten = Flatten()(pool4)
    dense = Dense(256,activation="relu")(flatten)
    dense = Dense(256,activation="relu")(dense)
    outputs = Dense(2,activation="softmax")(dense)
    model = Model(inputs=inputs,outputs=outputs)
    print(model.summary())
    sgd = optimizers.SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model

img_paths = get_img_path(dir_list)
# model = create_vgg16(960,960,1)
model = custom_net(28,28,1)
# early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', verbose=2)
model.fit_generator(generator=generate_arrays_from_file(img_paths[:3000]),steps_per_epoch=500,epochs=3)
