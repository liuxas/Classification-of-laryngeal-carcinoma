import os

import cv2
import keras
import numpy as np
import tensorflow as tf
from ipdb import set_trace
from keras import Input, Model, optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras.optimizers import SGD

print('Start build VGG16 -------')

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

dir_list = ["/home/liux/文档/项目/Classification-of-laryngeal-carcinoma/data/non_carcinoma/white_light/datagen_new/",
"/home/liux/文档/项目/Classification-of-laryngeal-carcinoma/data/carcinoma/white_light/datagen_new/"]
 
# 获取vgg16的卷积部分，如果要获取整个vgg16网络需要设置:include_top=True
def get_vgg_pre():
    model_vgg16_conv = VGG16(weights="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False)

    input_shape = (224, 224, 3)
    input = Input(input_shape, name = 'image_input')
    output_vgg16_conv = model_vgg16_conv(input)

    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dense(2, activation='softmax', name='predictions')(x)
    my_model = Model(inputs=input, outputs=x)
    sgd = optimizers.SGD(lr=0.0001, decay=1e-8, momentum=0.9, nesterov=True)
    my_model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    return my_model

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
            img = cv2.resize(img,(224,224))
            # print(img.shape)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[np.newaxis,:,:,:]
            img = img/255
            if "non" in img_path:
                label= np.array([[1,0]]).astype("int8")
            else:
                label= np.array([[0,1]]).astype("int8")  
            yield (img,label)

img_paths = get_img_path(dir_list)
model = get_vgg_pre()
model.fit_generator(generator=generate_arrays_from_file(img_paths[:3500]),steps_per_epoch=1500,epochs=50,
validation_data=generate_arrays_from_file(img_paths[3500:]),validation_steps=200,shuffle=True)
set_trace()
