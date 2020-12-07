import cv2
import keras
import numpy as np
import tensorflow as tf
# from ipdb import set_trace

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2040)])

img_path = "/home/liux/文档/项目/Classification-of-laryngeal-carcinoma/data/carcinoma/white_light/total_new/596509123425_4.jpg"
model_path = "./best_model.h5"
img = cv2.imread(img_path)
img = img.astype("float32")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img[np.newaxis,:,:,np.newaxis]
img = img/255

model = keras.models.load_model(model_path)
pred = model.predict(img)
pred_label = np.argmax(pred) #label为1，则为患有癌症
# set_trace()