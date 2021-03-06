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
from ipdb import set_trace

early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2040)])

dir_list = ["/home/liux/文档/项目/Classification-of-laryngeal-carcinoma/data/non_carcinoma/white_light/datagen_new/",
"/home/liux/文档/项目/Classification-of-laryngeal-carcinoma/data/carcinoma/white_light/datagen_new/"]

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
            img = cv2.imread(img_path)
            img = img.astype("float32")
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[np.newaxis,:,:,np.newaxis]
            img = img/255
            if "non" in img_path:
                label= np.array([[1,0]]).astype("int8")
            else:
                label= np.array([[0,1]]).astype("int8")  
            yield (img,label)


# def get_train_test():
#     data_list = []
#     for i in dir_list:
#         data_list.append(data_generator(i))
#     tran_test_data = np.append(data_list[0][0],data_list[1][0],axis=0)
#     labels = (np.append(data_list[0][1],data_list[1][1],axis=0))
#     labels = keras.utils.to_categorical(labels,num_classes=2)
#     index = (np.arange(labels.shape[0]))
#     #打乱顺序
#     np.random.shuffle(index)
#     tran_test_data = tran_test_data[index]
#     labels = labels[index]
#     train_num = int(0.9*labels.shape[0])
#     train_x = tran_test_data[0:train_num,:,:,:]
#     train_y = labels[0:train_num,:]
#     test_x = tran_test_data[train_num:,:,:,:]
#     test_y = labels[train_num:,:]
#     return train_x,train_y,test_x,test_y

def create_vgg16(img_h,img_w,img_ch):
    inputs = Input(shape=(img_h,img_w,img_ch))
    conv1 = Conv2D(64,3,activation="relu",padding="same")(inputs)
    # conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32,3,activation="relu",padding="same")(conv1)
    pool1 = MaxPooling2D(2)(conv1)

    conv2 = Conv2D(64,3,activation="relu",padding="same")(pool1)
    # conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64,3,activation="relu",padding="same")(conv2)
    pool2 = MaxPooling2D(2)(conv2)

    conv3 = Conv2D(128,3,activation="relu",padding="same")(pool2)
    # conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128,3,activation="relu",padding="same")(conv3)
    conv3 = Conv2D(128,1,activation="relu",padding="same")(conv3)
    pool3 = MaxPooling2D(2)(conv3)

    conv4 = Conv2D(256,3,activation="relu",padding="same")(pool3)
    # conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256,3,activation="relu",padding="same")(conv4)
    conv4 = Conv2D(256,1,activation="relu",padding="same")(conv4)
    pool4 = MaxPooling2D(2)(conv4)

    conv5 = Conv2D(256,3,activation="relu",padding="same")(pool4)
    conv5 = Conv2D(256,3,activation="relu",padding="same")(conv5)
    conv5 = Conv2D(256,1,activation="relu",padding="same")(conv5)
    pool5 = MaxPooling2D(2)(conv5)
    
    flatten = Flatten()(pool5)
    dense = Dense(256,activation="relu")(flatten)
    # dense = Dropout(0.2)(dense)
    dense = Dense(256,activation="relu")(dense)
    outputs = Dense(2,activation="softmax")(dense)

    model = Model(inputs=inputs,outputs=outputs)
    print(model.summary())

    sgd = optimizers.SGD(lr=0.0001, decay=1e-8, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    
    return model

def get_unet(img_height,img_width,img_ch):
    inputs = Input(shape=(img_height,img_width,img_ch))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(2)(conv3)

    conv4 = Conv2D(256,(3,3),activation="relu",padding="same")(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256,(3,3),activation="relu",padding="same")(conv4)
    pool4 = MaxPooling2D(2)(conv4)

    conv5 = Conv2D(512,(3,3),activation="relu",padding="same")(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(512,(3,3),activation="relu",padding="same")(conv5)

    up1 = UpSampling2D(size=(2, 2))(conv5)
    up1 = concatenate([conv4,up1],axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up1)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    #
    up2 = UpSampling2D(size=(2, 2))(conv6)
    up2 = concatenate([conv3,up2], axis=3)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    up3 = UpSampling2D(size=(2, 2))(conv7)
    up3 = concatenate([conv2,up3], axis=3)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up3)
    conv8 = Dropout(0.2)(conv7)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)
   
    conv9 = Conv2D(16,(1,1),activation="relu",padding="same")(conv8)
    conv9 = Conv2D(16,(1,1),strides=(2,2),activation="relu")(conv9)
    pool9 = MaxPooling2D(2)(conv9)

    flatten = Flatten()(pool9)
    dense = Dense(256,activation="relu")(flatten)
    dense = Dense(256,activation="relu")(dense)
    # dense = Dropout(0.2)(dense)
    outputs = Dense(2,activation="softmax")(dense)

    model = Model(inputs=inputs,outputs=outputs)
    print(model.summary())

    sgd = optimizers.SGD(lr=0.0001, decay=1e-8, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    return model

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
    sgd = optimizers.SGD(lr=0.0001, decay=1e-8, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model

img_paths = get_img_path(dir_list)
# model = create_vgg16(960,960,1)
model = custom_net(960,960,1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, mode='max', verbose=2)
model.fit_generator(generator=generate_arrays_from_file(img_paths[:3500]),steps_per_epoch=1500,epochs=50,
validation_data=generate_arrays_from_file(img_paths[3500:]),validation_steps=200,shuffle=True)
set_trace()
