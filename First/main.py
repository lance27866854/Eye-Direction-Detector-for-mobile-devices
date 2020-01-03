import tensorflow as tf

import cv2
from matplotlib import pyplot as plt
from collections import defaultdict
import numpy as np
import sys
import os
import time
import random
random.seed(5487)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from model import Model
from utils import document, plot_doc, write_info, write_test

############################
#           Flags          #
############################
# --------- Mode --------- #
tf.app.flags.DEFINE_boolean("is_train", True, "training mode.")
tf.app.flags.DEFINE_boolean("restore", False, "training mode.")
tf.app.flags.DEFINE_boolean("data_augmentation", True, "training mode.")
# ---- Hyperparameters --- #
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate.")
tf.app.flags.DEFINE_integer("epoch", 100, "Number of epoch.")
tf.app.flags.DEFINE_integer("batch", 8, "Number of batches(20*videos).")
tf.app.flags.DEFINE_integer("slice", 2000, "Number of images.") # small batch
# ------ Save Model ------ #
tf.app.flags.DEFINE_string("data_dir", "../dataset/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Weights directory")
tf.app.flags.DEFINE_string("test_dir", "test_known", "Test directory")
tf.app.flags.DEFINE_integer("per_checkpoint", 2000, "How many steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("inference_version", -1, "The version for inferencing.")
tf.app.flags.DEFINE_integer("val_num", 30000, "number of validation images.")

FLAGS = tf.app.flags.FLAGS

############################
#         Functions        #
############################

def get_data(is_train=FLAGS.is_train):
    in_1 = [] # (4-d tensor) shape : n, w, h, 3
    in_2 = [] # (4-d tensor) shape : n, w, h, 3
    in_3 = [] # (4-d tensor) shape : n, w, h, 3
    gt = []
    
    aug_dir = '/aug' if FLAGS.data_augmentation else '/no_aug'

    if is_train:
        for i in range(FLAGS.batch):
            print("Getting the "+str(i)+"-th batch.")
            data_ = np.load(FLAGS.data_dir+str(i)+aug_dir+'/first_data.npy', allow_pickle=True)
            in_1_, in_2_, in_3_, gt_ = data_[0], data_[1], data_[2], data_[3]
            # shuffle
            shuffle_index = np.arange(in_1_.shape[0])
            np.random.shuffle(shuffle_index)
            # append
            for j in range(in_1_.shape[0]):
                idx = shuffle_index[j]
                in_1.append(in_1_[idx])
                in_2.append(in_2_[idx])
                in_3.append(in_3_[idx])
                gt.append(gt_[idx])

    else:
        print("Getting the test .")
        data_ = np.load(FLAGS.data_dir+FLAGS.test_dir+aug_dir+'/first_data.npy', allow_pickle=True)
        in_1_, in_2_, in_3_, gt_ = data_[0], data_[1], data_[2], data_[3]
        for j in range(in_1_.shape[0]):
            in_1.append(in_1_[j])
            in_2.append(in_2_[j])
            in_3.append(in_3_[j])
            gt.append(gt_[j])

    return in_1, in_2, in_3, gt

def train(model, sess, data_1, data_2, data_3, gt):
    loss = [0,0,0]
    acc = [0,0,0]
    st, ed, times = 0, 0, 0
    max_len = len(data_1)

    # for every slice
    while st < max_len:
        ed = st + FLAGS.slice if st + FLAGS.slice < max_len else max_len
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

def validation(model, sess, data_1, data_2, data_3, gt):
    loss = [0,0,0]
    acc = [0,0,0]
    st, ed, times = 0, 0, 0
    max_len = len(data_1)

    # for every slice
    while st < max_len:
        ed = st + FLAGS.slice if st + FLAGS.slice < max_len else max_len
        feed = {model.x_1: data_1[st:ed], model.x_2: data_2[st:ed], model.x_3: data_3[st:ed], model.y: gt[st:ed]}
        loss_1, acc_1, loss_2, acc_2, loss_3, acc_3 = sess.run([model.val_loss_1, model.val_acc_1, model.val_loss_2, model.val_acc_2, model.val_loss_3, model.val_acc_3], feed_dict=feed)
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

def test(model, sess, data_1, data_2, data_3, gt, img_show=False):
    length = len(data_1)
    count = 0
    left_total = 0
    right_total = 0
    left_correct = 0
    right_correct = 0
    print(length)
    avg_time = 0

    for i in range(length):
        if (i%1000 == 999):
            print("processing the "+str(i+1)+"-th image.")

        if gt[i] == 1:
            left_total += 1
        elif gt[i] == 2:
            right_total += 1

        pred = [0, 0, 0]
        logit = [0, 0, 0]
        start_time = time.time()
        feed = {model.x_1: [data_1[i]], model.x_2: [data_2[i]], model.x_3: [data_3[i]], model.y: [gt[i]]}
        pred[0], pred[1], pred[2], logit[0], logit[1], logit[2] = sess.run([model.val_pred_1, model.val_pred_2, model.val_pred_3, model.val_logit_1, model.val_logit_2, model.val_logit_3], feed)
        avg_time += (time.time() - start_time)
        max_logit = np.argmax(np.array([max(logit[0][0]), max(logit[1][0]), max(logit[2][0])]))

        if pred[max_logit] == gt[i]:
            count += 1
            if gt[i] == 1:
                left_correct += 1
            elif gt[i] == 2:
                right_correct += 1

            if img_show and (gt[i] == 1 or gt[i] == 2):
                fig,ax = plt.subplots(1)
                ax.imshow(data_1[i])
                plt.title(str(gt[i]))
                plt.show()
                
    print(avg_time/length)
    return count/length, left_correct/left_total, right_correct/right_total

############################
#           Main           #
############################

# ------- variables ------ #
best_acc = 0
best_epoch = 0

# -------- config -------- #
config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True

# ------- run sess ------- #
with tf.Session(config=config) as sess:
    if FLAGS.is_train:
        
        # building computational graph (model)
        if FLAGS.restore:
            # inference version cannot be -1.
            cnn_model = Model(learning_rate=FLAGS.learning_rate)
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)
            cnn_model.saver.restore(sess, model_path)
        else:    
            cnn_model = Model(learning_rate=FLAGS.learning_rate)
            tf.global_variables_initializer().run()
        
        # train list (for documenting loss and acc.)
        doc = defaultdict(list)
        
        # get training data
        data_1, data_2, data_3, gt = get_data()
        print(len(gt))

        # get validation data
        data_val_1, data_val_2, data_val_3, gt_val = data_1[:FLAGS.val_num], data_2[:FLAGS.val_num], data_3[:FLAGS.val_num], gt[:FLAGS.val_num]
        data_1, data_2, data_3, gt = data_1[FLAGS.val_num:], data_2[FLAGS.val_num:], data_3[FLAGS.val_num:], gt[FLAGS.val_num:]

        for epoch in range(FLAGS.epoch):
            start_time = time.time()
            train_acc, train_loss = train(cnn_model, sess, data_1, data_2, data_3, gt)
            val_acc, val_loss = validation(cnn_model, sess, data_val_1, data_val_2, data_val_3, gt_val)

            acc = (val_acc[0]+val_acc[1]+val_acc[2])/3
            if acc >= best_acc:
                best_acc = acc
                best_epoch = epoch + 1
                cnn_model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, global_step=cnn_model.global_step)

            epoch_time = time.time() - start_time
            document(epoch, epoch_time, train_loss, train_acc, val_loss, val_acc, doc)
            
        plot_doc(doc)
        write_info(best_epoch, best_acc)
    
    else:
        cnn_model = Model()
        if FLAGS.inference_version == -1:
            print("Please set the inference version!")
        else:
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)

        cnn_model.saver.restore(sess, model_path)
        data_1, data_2, data_3, gt = get_data()
        test_acc, left_acc, right_acc = test(cnn_model, sess, data_1, data_2, data_3, gt)
        write_test(test_acc, left_acc, right_acc)
