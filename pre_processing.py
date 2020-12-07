import os

import cv2
import tensorflow as tf
from ipdb import set_trace
from keras.preprocessing.image import (ImageDataGenerator, img_to_array,
                                       load_img)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

# datagen = ImageDataGenerator(
#     rotation_range=20,#旋转范围, 随机旋转(0-180)度
#     width_shift_range=0.2,#随机沿着水平或者垂直方向，以图像的长宽小部分百分比为变化范围进行平移;
#     height_shift_range=0.2,
#     shear_range=0.2,#水平或垂直投影变换
#     zoom_range=0.2,#按比例随机缩放图像尺寸
#     horizontal_flip=True,#水平翻转图像
#     fill_mode='nearest')#填充像素, 出现在旋转或平移之后

datagen = ImageDataGenerator(
    rotation_range=20,#旋转范围, 随机旋转(0-180)度
    zoom_range=0.2,#按比例随机缩放图像尺寸
    horizontal_flip=True)#水平翻转图像

dir_list = ["/home/liux/文档/项目/Classification-of-laryngeal-carcinoma/data/non_carcinoma/white_light/total_new/",
"/home/liux/文档/项目/Classification-of-laryngeal-carcinoma/data/carcinoma/white_light/total_new/"]


def data_generate(img_dir):
    img_path = [img_dir+i for i in os.listdir(img_dir)]
    for i in img_path:
        j=0
        img=load_img(i)
        img = img_to_array(img)
        img = img.reshape((1,)+img.shape)
        for batch in datagen.flow(img,batch_size=1,save_to_dir='/home/liux/文档/项目/Classification-of-laryngeal-carcinoma/data/carcinoma/white_light/datagen_new',save_prefix="fail",save_format="jpg"):
            j+=1
            if j>10:
                break
    return None

data_generate(dir_list[1])
