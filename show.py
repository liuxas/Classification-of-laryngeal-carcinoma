import os

import cv2
import numpy as np
from ipdb import set_trace
from matplotlib import pyplot as plt

# img_path = "/home/liux/文档/项目/Classification-of-laryngeal-carcinoma/data/carcinoma/white_light/total/596517290691_5.jpg"
img_dir = "/home/liux/文档/项目/Classification-of-laryngeal-carcinoma/data/non_carcinoma/white_light/total/"
img_dir_new ="/home/liux/文档/项目/Classification-of-laryngeal-carcinoma/data/non_carcinoma/white_light/total_new/"

def preprosess_img(img_dir):
    imgs_path = [img_dir+i for i in os.listdir(img_dir)]
    # axis_0 = []
    # axis_1 = []
    # for i in imgs_path:
    #     img = cv2.imread(i) #读入一张图片，返回一个（H，W，C）的ndarray
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     axis_0.append(img.shape[0])
    #     axis_1.append(img.shape[1])
    # min_axis_0,min_axis_1 = min(axis_0),min(axis_1) # min_axis_0 = 716, min_axis_1 = 682

    for i in imgs_path:
        name = i.split("/")[-1]
        img = cv2.imread(i) #读入一张图片，返回一个（H，W，C）的ndarray
        set_trace()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(960,960))
        cv2.imwrite(img_dir_new+name, img)
    return None

preprosess_img(img_dir)
set_trace()

# cv2.imshow("non",img)	#显示一张图片
# cv2.waitKey()		#必须加上，不然图片无法显示
# cv2.destroyAllWindows()	#执行键盘动作后关闭窗口
