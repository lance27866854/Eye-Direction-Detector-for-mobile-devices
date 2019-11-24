import tensorflow as tf
from tensorflow.python.framework import constant_op
from collections import defaultdict

import numpy as np
import sys
import os
import time
import random
random.seed(5487)

from load_data import point_convertor, load_data, get_raw_points
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
# ----- for training ----- #
best_acc = 0
best_epoch = 0

############################
#           Flags          #
############################
# --------- Mode --------- #
tf.app.flags.DEFINE_integer("mode", 3, "training mode.")
# ---- Hyperparameters --- #
tf.app.flags.DEFINE_float("learning_rate", 0.005, "learning rate.")
tf.app.flags.DEFINE_integer("epoch", 10, "Number of epoch.")
tf.app.flags.DEFINE_integer("batch_size", 150, "Number of batches(videos).")
# ------ Save Model ------ #
tf.app.flags.DEFINE_string("data_dir", "../tool/dataset", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Data directory")
tf.app.flags.DEFINE_integer("per_checkpoint", 1000, "How many steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("inference_version", 96, "The version for inferencing.")

FLAGS = tf.app.flags.FLAGS

############################
#         Functions        #
############################
def mode_convertor(mode=0):
    # Mode 0: Training and also store the data points (usually for the first time).
    # Mode 1: Training but use stored data points (usually for the second time).
    # Mode 2: Testing for checking the accuracy (usually for the first time).
    # Mode 3: Testing for generate data for the cascaded network (usually for the second time).
    # return is_train, store_data_point, store_cascaded_data 
    if mode == 0:
        return True, True, False
    elif mode == 1:
        return True, False, False
    elif mode == 2:
        return False, False, False
    elif mode == 3:
        return False, False, True
    else:
        return False, True, False

def get_input(center_in, frame):
    in_Y_cut, in_X_cut = center_in[0], center_in[1]
    padding_frame = np.zeros(shape=(H_3*2+HEIGHT, W_3*2+WIDTH, 3))
    padding_frame[H_3 : H_3+HEIGHT, W_3 : W_3+WIDTH] += frame

    st_Y_1 = (int)(in_Y_cut-H_1/2)+H_3
    ed_Y_1 = (int)(in_Y_cut+H_1/2)+H_3
    st_Y_2 = (int)(in_Y_cut-H_2/2)+H_3
    ed_Y_2 = (int)(in_Y_cut+H_2/2)+H_3
    st_Y_3 = (int)(in_Y_cut-H_3/2)+H_3
    ed_Y_3 = (int)(in_Y_cut+H_3/2)+H_3

    st_X_1 = (int)(in_X_cut-W_1/2)+W_3
    ed_X_1 = (int)(in_X_cut+W_1/2)+W_3
    st_X_2 = (int)(in_X_cut-W_2/2)+W_3
    ed_X_2 = (int)(in_X_cut+W_2/2)+W_3
    st_X_3 = (int)(in_X_cut-W_3/2)+W_3
    ed_X_3 = (int)(in_X_cut+W_3/2)+W_3
    
    return padding_frame[st_Y_1:ed_Y_1, st_X_1:ed_X_1], padding_frame[st_Y_2:ed_Y_2, st_X_2:ed_X_2], padding_frame[st_Y_3:ed_Y_3, st_X_3:ed_X_3]

def get_data(store_data_point=True):
    region_cet = None
    if (store_data_point):
        region_cet, trX, shape = point_convertor(FLAGS.data_dir)            
        NUM_V, HEIGHT, WIDTH = shape
        print("Got "+str(region_cet.shape[0])+" points!")
        print("The prediction regions are all stored in the directory...")
    else:
        trX, trY, teX, teY, shape = load_data(FLAGS.data_dir)
        NUM_V, HEIGHT, WIDTH = shape
        region_cet = np.load("../tool/dataset/data_points.npy", allow_pickle=True)
        print("retriving the data from the directory...")

    data_1 = []
    data_2 = []
    data_3 = []
    gt = []
    
    # shuffle (?)
    for i in range(region_cet.shape[0]):# n
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
is_train, store_data_point, store_cascaded_data = mode_convertor(FLAGS.mode)
error = not is_train and store_data_point and not store_cascaded_data
assert error == False

# ------- run sess ------- #
with tf.Session(config=config) as sess:

    if is_train:
        data_1, data_2, data_3, gt = get_data(store_data_point)
        
        # build comp graph.
        cnn_model = Model()
        tf.global_variables_initializer().run()
        
        # training
        for epoch in range(FLAGS.epoch):
            start_time = time.time()
            train_acc, train_loss = train(cnn_model, sess, data_1, data_2, data_3, gt)
            
            acc = (train_acc[0]+train_acc[1]+train_acc[2])/3
            if acc >= best_acc:
                best_acc = acc
                best_epoch = epoch + 1
                cnn_model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, global_step=cnn_model.global_step)

            epoch_time = time.time() - start_time
            print("Epoch " + str(epoch + 1) + " of " + str(FLAGS.epoch) + " took " + str(epoch_time) + "s")
            print('LOSS : {:.4f}, {:.4f}, {:.4f}'.format(train_loss[0], train_loss[1], train_loss[2]))
            print('ACC  : {:.4f}, {:.4f}, {:.4f}'.format(train_acc[0], train_acc[1], train_acc[2]))
    
    elif not store_cascaded_data:
        cnn_model = Model()
        if FLAGS.inference_version == 0:
            model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        else:
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)

        cnn_model.saver.restore(sess, model_path)
        data_1, data_2, data_3, gt = get_data(store_data_point)
        count = 0
        for i in range(len(data_1)):
            pred = [0, 0, 0]
            logit = [0, 0, 0]
            feed = {cnn_model.x_1: [data_1[i]], cnn_model.x_2: [data_2[i]], cnn_model.x_3: [data_3[i]], cnn_model.y: [gt[i]]}
            pred[0], pred[1], pred[2], logit[0], logit[1], logit[2] = sess.run([cnn_model.pred_1, cnn_model.pred_2, cnn_model.pred_3, cnn_model.logit_1, cnn_model.logit_2, cnn_model.logit_3], feed)
            max_logit = np.argmax(np.array([max(logit[0][0]), max(logit[1][0]), max(logit[2][0])]))
            if pred[max_logit] == gt[i]:
                count += 1

        test_acc = float(count) / len(data_1)
        print("test accuracy: {}".format(test_acc))

    else :
        cnn_model = Model()
        if FLAGS.inference_version == 0:
            model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        else:
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)
        cnn_model.saver.restore(sess, model_path)

        print("Getting the raw data...")
        region_cet = None # shape : [video, frame, n, 2]
        region_cet, trX, shape = get_raw_points(FLAGS.data_dir)
        NUM_V, HEIGHT, WIDTH = shape
        data_region_cutter = [] # shape : [video, frame, 2, H, W]
        
        for i in range(len(region_cet)):
            video_region_cutter = []
            for j in range(len(region_cet[i])):
                max_left_logit = 0
                max_right_logit = 0
                frame_region_cutter = [None, None]

                for k in range(len(region_cet[i][j])):
                    pred = [0, 0, 0]
                    logit = [0, 0, 0]
                    inputs = [None, None, None]
                    inputs[0], inputs[1], inputs[2] = get_input(region_cet[i][j][k], trX[i][j])

                    feed = {cnn_model.x_1: [inputs[0]], cnn_model.x_2: [inputs[1]], cnn_model.x_3: [inputs[2]]}
                    pred[0], pred[1], pred[2], logit[0], logit[1], logit[2] = sess.run([cnn_model.pred_1, cnn_model.pred_2, cnn_model.pred_3, cnn_model.logit_1, cnn_model.logit_2, cnn_model.logit_3], feed)
                    logit = [max(logit[0][0]), max(logit[1][0]), max(logit[2][0])]
                    max_logit = np.argmax(np.array(logit))
                    
                    if pred[max_logit] == 1 and logit[max_logit] > max_left_logit: # left
                        max_left_logit = logit[max_logit]
                        frame_region_cutter[0] = cv2.resize(inputs[max_logit], dsize=(H_1, W_1), interpolation=cv2.INTER_AREA)
                        print("Video [{:d}] Frame [{:d}]: GOT left!".format(i, j))

                    elif pred[max_logit] == 2 and logit[max_logit] > max_right_logit: # right
                        max_right_logit = logit[max_logit]
                        frame_region_cutter[1] = cv2.resize(inputs[max_logit], dsize=(H_1, W_1), interpolation=cv2.INTER_AREA)
                        print("Video [{:d}] Frame [{:d}]: GOT right!".format(i, j))
                
                video_region_cutter.append(frame_region_cutter)
            
            data_region_cutter.append(video_region_cutter)
        
        data = np.array(data_region_cutter)
        np.save("../tool/dataset_RNN/batch0.npy", data)