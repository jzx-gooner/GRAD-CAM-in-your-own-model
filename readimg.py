# -*- coding: utf-8 -*-
from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
#resize
w=64
h=64
# c=3

def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    print(cate)
    imgs=[]
    labels=[]
    name=[]
    for idx,folder in enumerate(cate):
        for i in glob.glob(folder+'/*.JPG'):
            print('reading the images:%s'%(i))
	    name.append(i)
            img=cv2.imread(i)
            img=cv2.resize(img,(w,h), cv2.INTER_LINEAR)
            labels.append(idx)
            HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#灰度图像
            edges = cv2.Canny(gray,50,150)

            #**********hough_channel*********
            hough_channel = np.zeros(gray.shape, np.uint8)
            lines = cv2.HoughLines(edges,1,np.pi/180,10)  #
            try:
                for rho,theta in lines[0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    cv2.line(hough_channel,(x1,y1),(x2,y2),(255),1)
            except Exception as e:
                print 'There is no lines to be detected!'

            #********Sobel边缘检测************
            sobelX = cv2.Sobel(gray,cv2.CV_64F,1,0)#x方向的梯度
            sobelY = cv2.Sobel(gray,cv2.CV_64F,0,1)#y方向的梯度
            sobelX = np.uint8(np.absolute(sobelX))#x方向梯度的绝对值
            sobelY = np.uint8(np.absolute(sobelY))#y方向梯度的绝对值

            #********fast角点提取*******
            fast_channel = np.zeros(gray.shape, np.uint8)
            fast=cv2.FastFeatureDetector_create(threshold=2,nonmaxSuppression=True,type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)#获取FAST角点探测器
            kp=fast.detect(img,None)#描述符
            fast_channel = cv2.drawKeypoints(gray,kp,fast_channel,color=(255))#
            fast_channel = cv2.cvtColor(fast_channel,cv2.COLOR_BGR2GRAY)

            #***********H，s两个通道*******
            HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            H, S, V = cv2.split(HSV)

            #*******hsv过滤***********
            Lower = np.array([0, 0, 0])
            Upper = np.array([255,33, 255])
            mask = cv2.inRange(HSV, Lower, Upper)
            hsv_channel = cv2.bitwise_and(img, img, mask=mask)
            hsv_channel = cv2.cvtColor(hsv_channel,cv2.COLOR_BGR2GRAY)

            #resize
            #img=cv2.resize(img,(64,64),interpolation=cv2.INTER_CUBIC)
            #hough_channel=cv2.resize(hough_channel,(100,100),interpolation=cv2.INTER_CUBIC)
            # sobelX=cv2.resize(sobelX,(w,h),interpolation=cv2.INTER_CUBIC)
            # sobelY=cv2.resize(sobelY,(w,h),interpolation=cv2.INTER_CUBIC)
            #fast_channel = cv2.resize(fast_channel,(100,100),interpolation=cv2.INTER_CUBIC)
            #hsv_channel = cv2.resize(hsv_channel,(64,64),interpolation=cv2.INTER_CUBIC)
            # gray=cv2.resize(gray,(64,64),interpolation=cv2.INTER_CUBIC)
            #H_channel=cv2.resize(H,(w,h),interpolation=cv2.INTER_CUBIC)
            #S_channel=cv2.resize(S,(w,h),interpolation=cv2.INTER_CUBIC)

            ##图像-均值 to-do:improve speed
            # sobleX=sobelX-sobelX.mean()
            # sobelY=sobelY-sobelY.mean()
            # H_channel=H_channel-H_channel.mean()
            # S_channel=S_channel-S_channel.mean()
            # print("计算完均值")

            #imshow
            # plt.subplot(161),plt.imshow(img,),plt.title('RGB')
            # plt.subplot(162),plt.imshow(hough_channel,),plt.title('hough')
            # plt.subplot(163),plt.imshow(sobelX,),plt.title('x')
            # plt.subplot(164),plt.imshow(sobelY,),plt.title('y')
            # plt.subplot(165),plt.imshow(fast_channel),plt.title('fast')
            # plt.subplot(165),plt.imshow(hsv_channel),plt.title('hsv')
            # plt.show()
            #plt.savefig("./savefig/examples.jpg")

            #转换为输入矩阵 [img,gray,sobelX,sobelY,H,S,hsv_channel,fast_channel,hough_channel]
            #     对应为   RGB，GRAY，X梯度  Y梯度   H通道，S通道，HSV过滤，fast提取，hough line
            #mergedByNp = np.dstack([gray,sobelX,sobelY])

            #
            #img = tf.image.per_image_standardization(img)
            r, g, b = cv2.split(img)
            #print(cv2.mean(r)[0])
            n_r=r-cv2.mean(r)[0]
            n_g=g-cv2.mean(g)[0]
            n_b=b-cv2.mean(b)[0]
            new_img=cv2.merge([n_r,n_g,n_b])
            
            mergedByNp = np.dstack([img])
            #N=mergedByNp.shape[-1]
            print(mergedByNp.shape)
            imgs.append(mergedByNp)

            #imgs.append(hsv_channel)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32),name
