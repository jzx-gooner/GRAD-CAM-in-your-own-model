#coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('/home/jzx/tijiao2/model/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('/home/jzx/tijiao2/model/'))
    graph = tf.get_default_graph()
    #conv
    conv1_Weights=(sess.run(tf.get_default_graph().get_tensor_by_name("layer1-conv1/weight:0")))
    print(conv1_Weights.shape)
    print(np.shape(conv1_Weights))
    conv1_biases = (sess.run(tf.get_default_graph().get_tensor_by_name("layer1-conv1/bias:0")))
    conv2_Weights=(sess.run(tf.get_default_graph().get_tensor_by_name("layer2-conv2/weight:0")))
    conv2_biases = (sess.run(tf.get_default_graph().get_tensor_by_name("layer2-conv2/bias:0")))
    conv3_Weights=(sess.run(tf.get_default_graph().get_tensor_by_name("layer4-conv3/weight:0")))
    conv3_biases = (sess.run(tf.get_default_graph().get_tensor_by_name("layer4-conv3/bias:0")))
    #fc
    fc1_Weights=(sess.run(tf.get_default_graph().get_tensor_by_name("layer6-fc1/weight:0")))
    fc1_biases = (sess.run(tf.get_default_graph().get_tensor_by_name("layer6-fc1/bias:0")))
    fc2_Weights=(sess.run(tf.get_default_graph().get_tensor_by_name("layer7-fc2/weight:0")))
    fc2_biases = (sess.run(tf.get_default_graph().get_tensor_by_name("layer7-fc2/bias:0")))

    # conv1_Weights=np.save("conv1_Weights", conv1_Weights)
    # conv1_biases = np.save("conv1_biases", conv1_biases)
    # conv2_Weights = np.save("conv2_Weights", conv2_Weights)
    # conv2_biases = np.save("conv2_biases", conv2_biases)
    # conv3_Weights = np.save("conv3_Weights", conv3_Weights)
    # conv3_biases = np.save("conv3_biases", conv3_biases)
    # fc1_Weights = np.save("fc1_Weights", fc1_Weights)
    # fc1_biases = np.save("fc1_biases", fc1_biases)
    # fc2_Weights = np.save("fc2_Weights", fc2_Weights)
    # fc2_biases = np.save("fc2_biases", fc2_biases)
    np.savez("model_weights.npz", conv1_Weights=conv1_Weights,conv1_biases=conv1_biases,conv2_Weights=conv2_Weights,conv2_biases=conv2_biases,conv3_Weights=conv3_Weights,conv3_biases=conv3_biases,fc1_Weights=fc1_Weights,fc1_biases=fc1_biases,fc2_Weights=fc2_Weights,fc2_biases=fc2_biases)
    print("meta to npz is done")
    D = np.load("model_weights.npz")
    print(D["conv1_Weights"].shape)
