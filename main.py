import tensorflow as tf
from tensorflow.python.framework import constant_op
from collections import defaultdict

import numpy as np
import sys
import os
import time
import random
random.seed(5487)

from load_data import point_convertor, load_data
from model import Model

import cv2
from matplotlib import pyplot as plt

############################
#         Parameters       #
############################
# ------ kernel size ----- #
W_1 = 140
H_1 = 70
W_2 = 160
H_2 = 80
W_3 = 180
H_3 = 90
# ------- video size ----- #
NUM_V = 0
HEIGHT = 960
WIDTH = 544

############################
#           Flags          #
############################
tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_float("learning_rate", 0.005, "learning rate.")
tf.app.flags.DEFINE_integer("epoch", 10, "Number of epoch.")
tf.app.flags.DEFINE_integer("batch_size", 15, "Number of batches(videos).")
tf.app.flags.DEFINE_string("data_dir", "./tool/dataset", "Data directory")
tf.app.flags.DEFINE_integer("per_checkpoint", 1000, "How many steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("inference_version", 0, "The version for inferencing.")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")
tf.app.flags.DEFINE_boolean("store_data_points", False, "Set to True to show the parameters")
FLAGS = tf.app.flags.FLAGS

############################
#         Functions        #
############################
def get_input(center_in, frame):
    in_Y_cut, in_X_cut = center_in[0], center_in[1]
    
    st_Y_1 = (int)(in_Y_cut-H_1/2) if in_Y_cut >= H_1/2 else 0
    ed_Y_1 = (int)(in_Y_cut+H_1/2) if in_Y_cut+H_1/2 <= HEIGHT else HEIGHT
    st_Y_2 = (int)(in_Y_cut-H_2/2) if in_Y_cut >= H_2/2 else 0
    ed_Y_2 = (int)(in_Y_cut+H_2/2) if in_Y_cut+H_2/2 <= HEIGHT else HEIGHT
    st_Y_3 = (int)(in_Y_cut-H_3/2) if in_Y_cut >= H_3/2 else 0
    ed_Y_3 = (int)(in_Y_cut+H_3/2) if in_Y_cut+H_3/2 <= HEIGHT else HEIGHT

    st_X_1 = (int)(in_X_cut-W_1/2) if in_X_cut>=W_1/2 else 0
    ed_X_1 = (int)(in_X_cut+W_1/2) if in_X_cut+W_1/2<=WIDTH else WIDTH
    st_X_2 = (int)(in_X_cut-W_2/2) if in_X_cut>=W_2/2 else 0
    ed_X_2 = (int)(in_X_cut+W_2/2) if in_X_cut+W_2/2<=WIDTH else WIDTH
    st_X_3 = (int)(in_X_cut-W_3/2) if in_X_cut>=W_3/2 else 0
    ed_X_3 = (int)(in_X_cut+W_3/2) if in_X_cut+W_3/2<=WIDTH else WIDTH
    
    return frame[st_Y_1:ed_Y_1, st_X_1:ed_X_1], frame[st_Y_2:ed_Y_2, st_X_2:ed_X_2], frame[st_Y_3:ed_Y_3, st_X_3:ed_X_3]

def get_data():
    region_cet = None
    if (FLAGS.store_data_points):
        region_cet, trX, shape = point_convertor(FLAGS.data_dir, True)
        NUM_V, HEIGHT, WIDTH = shape
    else: 
        trX, trY, teX, teY, shape = load_data(FLAGS.data_dir)
        NUM_V, HEIGHT, WIDTH = shape
        region_cet = np.load("tool/dataset/data_points.npy", allow_pickle=True)
        
    # shuffle (?)
    data_1 = []
    data_2 = []
    data_3 = []
    gt = []
    for i in range(region_cet.shape[0]):
        center_in, center_gt, video_idx, frame_idx = region_cet[i][0], region_cet[i][1], region_cet[i][2], region_cet[i][3] # shape : [n, (2, 1, 1, 1)]
        in_1, in_2, in_3 = get_input(center_in, trX[video_idx][frame_idx])
        data_1.append(in_1)
        data_2.append(in_2)
        data_3.append(in_3)
        gt.append(center_gt)

    return data_1, data_2, data_3, gt

def train(model, sess, data_1, data_2, data_3, gt):
    loss = [0,0,0]
    acc = [0,0,0]
    st, ed, times = 0, 0, 0
    max_len = len(data_1)
    # for every batch
    while st < max_len:
        ed = st + FLAGS.batch_size if st + FLAGS.batch_size < max_len else max_len
        feed = {model.x_1: data_1[st:ed], model.x_2: data_2[st:ed], model.x_3: data_3[st:ed], model.y: gt[st:ed]}
        loss_1, acc_1, loss_2, acc_2, loss_3, acc_3, _1, _2, _3 = sess.run([model.loss_1, model.acc_1, model.loss_2, model.acc_2, model.loss_3, model.acc_3, model.train_op1, model.train_op2, model.train_op3], feed_dict=feed)
        loss[0] += loss_1
        loss[1] += loss_2
        loss[2] += loss_3
        acc[0] += acc_1
        acc[1] += acc_2
        acc[2] += acc_3
        times += 1
        st = ed

    loss[0] /= times
    loss[1] /= times
    loss[2] /= times
    acc[0] /= times
    acc[1] /= times
    acc[2] /= times
    return acc, loss


############################
#           Main           #
############################

# -------- config -------- #
config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True

# ------- run sess ------- #
with tf.Session(config=config) as sess:

    if FLAGS.is_train:
        data_1, data_2, data_3, gt = get_data()
        
        # build comp graph.
        cnn_model = Model()
        tf.global_variables_initializer().run()
        
        # training
        for epoch in range(FLAGS.epoch):
            start_time = time.time()
            train_acc, train_loss = train(cnn_model, sess, data_1, data_2, data_3, gt)
            epoch_time = time.time() - start_time
            print("Epoch " + str(epoch + 1) + " of " + str(FLAGS.epoch) + " took " + str(epoch_time) + "s")
            print('LOSS : {:.4f}, {:.4f}, {:.4f}'.format(train_loss[0], train_loss[1], train_loss[2]))
            print('ACC  : {:.4f}, {:.4f}, {:.4f}'.format(train_acc[0], train_acc[1], train_acc[2]))
        