import tensorflow as tf
from tensorflow.python.framework import constant_op
from collections import defaultdict

import numpy as np
import sys
import os
import random
random.seed(5487)

from load_data import load_data, get_shape
from preprocessing import Video


############################
#         Parameters       #
############################
# ------ kernel size ----- #
W_1 = 100
H_1 = 50
W_2 = 120
H_2 = 60
W_3 = 140
H_3 = 70
# ------- video size ----- #
NUM_V = 0
HEIGHT = 0
WIDTH = 0

############################
#           Flags          #
############################
tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_float("learning_rate", 0.005, "learning rate.")
tf.app.flags.DEFINE_integer("epoch", 10, "Number of epoch.")
tf.app.flags.DEFINE_integer("batch_size", 2, "Number of batches(videos).")
tf.app.flags.DEFINE_string("data_dir", "./tool/dataset", "Data directory")
tf.app.flags.DEFINE_integer("per_checkpoint", 1000, "How many steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("inference_version", 0, "The version for inferencing.")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")
FLAGS = tf.app.flags.FLAGS

############################
#         Functions        #
############################
def get_input(center_in, center_idx, i, trX):
    in_Y_cut, in_X_cut = center_in[0], center_in[1]
    
    st_Y_1 = in_Y_cut-H_1/2 if in_Y_cut>=H_1/2 else 0
    ed_Y_1 = in_Y_cut+H_1/2 if in_Y_cut+H_1/2<=HEIGHT else HEIGHT
    st_Y_2 = in_Y_cut-H_2/2 if in_Y_cut>=H_2/2 else 0
    ed_Y_2 = in_Y_cut+H_2/2 if in_Y_cut+H_2/2<=HEIGHT else HEIGHT
    st_Y_3 = in_Y_cut-H_3/2 if in_Y_cut>=H_3/2 else 0
    ed_Y_3 = in_Y_cut+H_3/2 if in_Y_cut+H_3/2<=HEIGHT else HEIGHT

    st_X_1 = in_X_cut-W_1/2 if in_X_cut>=W_1/2 else 0
    ed_X_1 = in_X_cut+W_1/2 if in_X_cut+W_1/2<=WIDTH else WIDTH
    st_X_2 = in_X_cut-W_2/2 if in_X_cut>=W_2/2 else 0
    ed_X_2 = in_X_cut+W_2/2 if in_X_cut+W_2/2<=WIDTH else WIDTH
    st_X_3 = in_X_cut-W_3/2 if in_X_cut>=W_3/2 else 0
    ed_X_3 = in_X_cut+W_3/2 if in_X_cut+W_3/2<=WIDTH else WIDTH
    
    return trX[center_idx][i][st_Y_1:ed_Y_1][st_X_1:ed_X_1], trX[center_idx][i][st_Y_2:ed_Y_2][st_X_2:ed_X_2], trX[center_idx][i][st_Y_3:ed_Y_3][st_X_3:ed_X_3]

def train(model, sess, region_points, trX):
    loss, acc = 0.0, 0.0
    st, ed, times = 0, FLAGS.batch_size, 0
    # for every batch
    while st < len(region_points) and ed <= len(region_points):
        regions, labels = region_points[st:ed][0], region_points[st:ed][1]

        for i in len(regions):
            if len(regions[i]) == 0:
                continue
            center_in, center_gt, center_idx = regions[i][0], regions[i][1], regions[i][2] # shape : 2, 1, 1
            in_1, in_2, in_3 = get_input(center_in, center_idx, i, trX)
            feed = {model.x_1: in_1, model.x_2: in_2, model.x_3: in_3, model.y: center_gt}
            loss_, acc_, _ = sess.run([model.loss, model.acc, model.train_op], feed_dict=feed)
            loss += loss_
            acc += acc_
            # renew
            st, ed = ed, ed+FLAGS.batch_size
            times += 1

    loss /= times
    acc /= times
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
        trX, trY, teX, teY = load_data(FLAGS.data_dir)
        NUM_V, HEIGHT, WIDTH = get_shape(FLAGS.data_dir)
        region_cet = [] # shape : [# of video in batch, frames, # of points, 2]
        for i in range(len(trX)): # len(trX) -> all
            v = Video(trX[i], trY[i], i)
            region_points = v.get_candidate_regions() # shape : [frames, num_points, 2] -> n frame
            region_cet.append(region_points) # shape : [# of videos, frames, num_points, 2]
            print(region_points)

        # build comp graph.
        cnn_model = Model()
        if FLAGS.log_parameters:
            model.print_parameters()
        tf.global_variables_initializer().run()

        # init some parameters
        pre_losses = [1e18] * 3
        best_val_acc = 0.0

        for epoch in range(FLAGS.epoch):
            # training
            start_time = time.time()
            train_acc, train_loss = train(cnn_model, sess, region_points, trX)
            epoch_time = time.time() - start_time
            print("Epoch " + str(epoch + 1) + " of " + str(FLAGS.epoch) + " took " + str(epoch_time) + "s")
            print("  learning rate:                 " + str(cnn_model.learning_rate.eval()))
            print("  training loss:                 " + str(train_loss))
            print("  training accuracy:             " + str(train_acc))

            if train_loss > max(pre_losses):
                sess.run(cnn_model.learning_rate_decay_op)
            pre_losses = pre_losses[1:] + [train_loss]
        