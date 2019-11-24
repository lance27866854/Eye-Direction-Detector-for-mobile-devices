import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op

import cv2
import time
import os
import random
random.seed(5487)

from load_data import load_data
from model import Model

############################
#         Parameters       #
############################
# ----- for training ----- #
best_acc = 0
best_epoch = 0

############################
#           Flags          #
############################
# --------- Mode --------- #
tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
# ---- Hyperparameters --- #
tf.app.flags.DEFINE_float("learning_rate", 0.005, "Number of labels.")
tf.app.flags.DEFINE_integer("epoch", 20, "Number of epoch.")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size to use during training.")
# ------ Save Model ------ #
tf.app.flags.DEFINE_string("data_dir", "../tool/dataset", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_integer("per_checkpoint", 1000, "How many steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("inference_version", 0, "The version for inferencing.")

FLAGS = tf.app.flags.FLAGS

############################
#         Functions        #
############################
def train(model, sess, trX, trY): # shape : [video, frame, 2, H, W], video(batch size)
    loss, acc = 0.0, 0.0
    num_video = len(trX)

    for i in range(num_video):
        if trX[i][0] == None or trX[i][1] == None:
            continue
        feed = {model.x: [trX[i]], model.y: [trY[i]]}
        loss_, acc_, _ = sess.run([model.loss, model.acc, model.update], feed_dict=feed)
        loss += loss_
        accuracy += acc_

    return loss/num_video, accuracy/num_video

############################
#           Main           #
############################

# -------- config -------- #
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# ------- run sess ------- #
with tf.Session(config=config) as sess:

    if FLAGS.is_train:
        trX, trY, shape = load_data(FLAGS.data_dir, 0)
        
        rnn_model = Model(learning_rate=FLAGS.learning_rate)

        tf.global_variables_initializer().run()

        for epoch in range(FLAGS.epoch):
            start_time = time.time()
            train_loss, train_acc = train(rnn_model, sess, trX, trY)
            
            acc = (train_acc[0]+train_acc[1]+train_acc[2])/3
            if acc >= best_acc:
                best_acc = acc
                best_epoch = epoch + 1
                rnn_model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, global_step=rnn_model.global_step)

            epoch_time = time.time() - start_time
            print("Epoch " + str(epoch + 1) + " of " + str(FLAGS.epoch) + " took " + str(epoch_time) + "s")
            print('LOSS : {:.4f}, {:.4f}, {:.4f}'.format(train_loss[0], train_loss[1], train_loss[2]))
            print('ACC  : {:.4f}, {:.4f}, {:.4f}'.format(train_acc[0], train_acc[1], train_acc[2]))
