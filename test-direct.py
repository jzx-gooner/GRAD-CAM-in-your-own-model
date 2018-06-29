# -*- coding: utf-8 -*-
from __future__ import division
import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt

class_list=['blade','no','blade-tip']

path = "./19.JPG"

w=64
h=64
c=4

def read_img(path):
    img=cv2.imread(path)
    img=cv2.resize(img,(w,h), cv2.INTER_LINEAR)
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
    mergedByNp = np.dstack([gray,H,sobelX,sobelY])
    x=np.expand_dims(mergedByNp,axis=0)
    # Converting RGB to BGR for VGG
    print("the shape of x ")
    #x = x[:,:,:,::-1]
    print(x.shape)
    return x, mergedByNp,img

# def grad_cam(x, vgg, sess, predicted_class, layer_name, nb_classes):
# 	print("Setting gradients to 1 for target class and rest to 0")
# 	conv_layer = vgg.layers[layer_name]  #最后
# 	one_hot = tf.sparse_to_dense(predicted_class, [nb_classes], 1.0)
# 	signal = tf.multiply (vgg.layers['layer7-fc2'], one_hot)
# 	loss = tf.reduce_mean(signal)
# 	grads = tf.gradients(loss, conv_layer)[0]
# 	# Normalizing the gradients
# 	norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))
#
# 	output, grads_val = sess.run([conv_layer, norm_grads], feed_dict={vgg.imgs: x})
# 	output = output[0]
# 	grads_val = grads_val[0]
#
# 	weights = np.mean(grads_val, axis = (0, 1))
# 	cam = np.ones(output.shape[0 : 2], dtype = np.float32)
#
# 	# Taking a weighted average
# 	for i, w in enumerate(weights):
# 	    cam += w * output[:, :, i]
#
# 	# Passing through ReLU
# 	cam = np.maximum(cam, 0)
# 	cam = cam / np.max(cam)
# 	cam = cv2.resize(cam, (64,64))
#
# 	# Converting grayscale to 3-D
# 	cam3 = np.expand_dims(cam, axis=2)
# 	cam3 = np.tile(cam3,[1,1,3])
#
# 	return cam3


with tf.Session() as sess:
    data,_,img=read_img(path)
    #print(data)
    saver = tf.train.import_meta_graph('./model/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./model/'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}
    logits = graph.get_tensor_by_name('logits_eval:0')
    classification_result = sess.run(logits,feed_dict)
    # a=[tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    # print(a)
    #打印出预测矩阵
    print(classification_result)
    #打印出预测矩阵每一行最大值的索引
    print(tf.argmax(classification_result,1).eval())
    predicted_class=tf.argmax(classification_result,1).eval()
    #

    #获取预测结果的one-hot
    one_hot = tf.sparse_to_dense(predicted_class, [3], 1.0)
    #获得最后一层pool的tensor
    conv_layer = graph.get_tensor_by_name('layer5-pool2/layer5-pool2:0')
    print("11111111111111111")
    print(conv_layer.shape)
    #获取最后一层输出的tensor
    fc2 = graph.get_tensor_by_name('layer7-fc2/layer7-fc2:0')
    print("22222222222222222")
    print(fc2.shape)
    #fc2乘预测结果
    signal = tf.multiply(fc2, one_hot)

    #求均值作为loss
    loss = tf.reduce_mean(signal)
    #计算loss与最后一层的梯度
    grads = tf.gradients(loss, conv_layer)[0]
    # Normalizing the gradients
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

    output, grads_val = sess.run([conv_layer, norm_grads], feed_dict={x:data})
    
    output = output[0]
    grads_val = grads_val[0]
    print("grad_view")
    print(grads_val.shape)

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0: 2], dtype=np.float32)

    # Taking a weighted average
    for i, w in enumerate(weights):
	    cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = cv2.resize(cam, (64, 64))

    # Converting grayscale to 3-D
    cam3 = np.expand_dims(cam, axis=2)
    cam3 = np.tile(cam3, [1, 1, 3])

    cam3 = cv2.applyColorMap(np.uint8(255 * cam3), cv2.COLORMAP_JET)
    cam3 = cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB)
    cam3 = cv2.resize(cam3, (64, 64))

    img = cv2.resize(img, (64, 64)).astype(float)
    img /= img.max()
    #Superimposing the visualization with the image.am = cv2.resize(cam, (self.result_size, self.result_size))
    # print(img.shape)
    new_img = img+0.0025*cam3
    new_img /= new_img.max()

    # Display and save
    plt.subplot(131),plt.imshow(img,),plt.title('img')
    #plt.subplot(142),plt.imshow(new_img,),plt.title('average-img')
    plt.subplot(132),plt.imshow(cam3,),plt.title('cam3')
    plt.subplot(133),plt.imshow(new_img,),plt.title('grad-cam')
    #plt.title('the predicted_class is :'+str(classname[int(preds)]),color='red')
    plt.savefig("test.jpg")
    plt.show()
    # io.imshow
    # plt.show()
    # io.imshow(new_img)
    # plt.show()
    # io.imsave(, new_img)


    #根据索引通过字典对应叶片的分类
    # print("aaaaaaaaaaaaaaa")
    # print(classification_result)
    # preds = (np.argsort(classification_result)[::-1])[0:1]
    # print("bbbbbbbbbbbb")
    # print(preds)
    # predicted_class = preds[0]
    # print(predicted_class)
