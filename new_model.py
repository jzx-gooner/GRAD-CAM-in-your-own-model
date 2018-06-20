########################################################################################
# jzx                                                                 #
# blade_model implementation in TensorFlow                                             #
########################################################################################

import tensorflow as tf
import numpy as np
from imagenet_classes1 import class_names
import cv2

class blade_model:
    
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc2l)  #
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []
        self.layers = {}
        # zero-mean input
        # with tf.name_scope('preprocess') as scope:
        #     mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        #     images = self.imgs-mean

        # layer1-conv1
        images=self.imgs
        print(images.shape)
        with tf.name_scope('layer1-conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 6], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[6], dtype=tf.float32),
                                 trainable=True, name='bias')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.layers[scope[:-1]] = self.conv1_1
            self.parameters += [kernel, biases]

        # layer2-conv2
        with tf.name_scope('layer2-conv2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 6, 12], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[12], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.layers[scope[:-1]] = self.conv1_2
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool1')
        self.layers['layer3-pool1'] = self.pool1

        # layer4-conv3
        with tf.name_scope('layer4-conv3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 12, 24], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[24], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.layers[scope[:-1]] = self.conv2_1
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='layer5-pool2')
        self.layers['layer5-pool2'] = self.pool2


    def fc_layers(self):
        # fc1
        with tf.name_scope('layer6-fc1') as scope:
            shape = int(np.prod(self.pool2.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 512],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool2_flat = tf.reshape(self.pool2, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool2_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.layers[scope[:-1]] = self.fc1
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('layer7-fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([512, 2],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[2], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.layers[scope[:-1]] = self.fc2l
            self.parameters += [fc2w, fc2b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        print(weights)
        keys = sorted(weights.keys())
        print(keys)
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

# if __name__ == '__main__':
#
#     sess = tf.Session()
#     imgs = tf.placeholder(tf.float32, [None, 64, 64, 3])
#     vgg = blade_model(imgs, 'model_weights.npz', sess)
#     print("123")
#
#     img1 = cv2.imread('laska.png')
#     img1 = cv2.resize(img1, (64, 64))
#     print(img1.shape)
#
#     prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
#     preds = (np.argsort(prob)[::-1])[0:5]
#     for p in preds:
#         print class_names[p], prob[p]
