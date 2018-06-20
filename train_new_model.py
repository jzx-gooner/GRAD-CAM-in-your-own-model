# -*- coding: utf-8 -*-

from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from readimg import read_img
from tensorflow.contrib.layers.python.layers import batch_norm


#数据集地址
path='./dataset/'
#模型保存地址
model_path='./model/model.ckpt'
w=64
h=64
c=3

data,label,_=read_img(path)
#打乱顺序
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
print(data.shape)
label=label[arr]
#将所有数据分为训练集和验证集 0-0.8 0.8-1
ratio=0.8
s1=np.int(num_example*ratio)
x_train=data[:s1]
y_train=label[:s1]
x_val=data[s1:]
y_val=label[s1:]



#-----------------构建网络----------------------

x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')
keep_prob = tf.placeholder(tf.float32)

#bn层
def batch_norm_layer(value,train=None,name="batch_norm"):
    if train is not None:
        return batch_norm(value,decay=0.9,updates_collections=None,is_training=True)
    else:
        return batch_norm(value,decay=0.9,updates_collections=None,is_training=False)

#nn
def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight",[3,3,3,6],initializer=tf.truncated_normal_initializer(stddev=1.0))
        conv1_biases = tf.get_variable("bias", [6], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        temp1=tf.nn.bias_add(conv1, conv1_biases)
        relu1 = tf.nn.relu(batch_norm_layer(temp1,train))

    with tf.variable_scope("layer2-conv2"):
        conv2_weights = tf.get_variable("weight",[3,3,6,12],initializer=tf.truncated_normal_initializer(stddev=1.0))
        conv2_biases = tf.get_variable("bias", [12], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(relu1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        temp2=tf.nn.bias_add(conv2, conv2_biases)
        relu2= tf.nn.relu(batch_norm_layer(temp2,train))

    with tf.name_scope("layer3-pool1"):
        pool1 = tf.nn.max_pool(relu2, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,beta=0.75,name='norm1')

    with tf.variable_scope("layer4-conv3"):
        conv3_weights = tf.get_variable("weight",[3,3,12,24],initializer=tf.truncated_normal_initializer(stddev=1.0))
        conv3_biases = tf.get_variable("bias", [24], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(norm1, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        temp3=tf.nn.bias_add(conv3, conv3_biases)
        relu3= tf.nn.relu(batch_norm_layer(temp3,train))

    with tf.name_scope("layer5-pool2"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #nodes=28*28*24
        nodes=16*16*24
	reshaped = tf.reshape(pool3,[-1,nodes])

    with tf.variable_scope('layer6-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer7-fc2'):
        fc2_weights = tf.get_variable("weight", [512, 2],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [2], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases


    return logit


#---------------------------网络结束---------------------------
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
logits = inference(x,False,regularizer)

#将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits,b,name='logits_eval')

loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)

train_op=tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

#训练和测试数据，可将n_epoch和batch_size
n_epoch=int(raw_input("输入epoch："))
batch_size=int(raw_input("输入batcha_size:"))
saver=tf.train.Saver()
#GPU和cpu选择
config = tf.ConfigProto(device_count = {'GPU': 1}) #0表示仅cpu，1表示使用gpu
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#tensorboard  tensorboard --logdir=./log
summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter("./log",sess.graph)
writer.add_graph(sess.graph)

#sess=tf.Session()

sess.run(tf.global_variables_initializer())

for epoch in range(n_epoch):
    start_time = time.time()
    #print(x.shape)
    print("====epoch %d====="%epoch)
    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _=sess.run([train_op],feed_dict={x: x_train_a, y_: y_train_a, keep_prob: 0.5})
        err,ac=sess.run([loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err; train_acc += ac; n_batch += 1
    print("   train loss: %f" % (np.sum(train_loss)/ n_batch))
    print("   train acc: %f" % (np.sum(train_acc)/ n_batch))
    #validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=True):
        err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err; val_acc += ac; n_batch += 1
    print("   validation loss: %f" % (np.sum(val_loss)/ n_batch))
    print("   validation acc: %f" % (np.sum(val_acc)/ n_batch))

#保存模型
saver.save(sess,model_path)
writer.close()
sess.close()
