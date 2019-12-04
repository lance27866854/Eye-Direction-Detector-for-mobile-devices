import numpy as np
import tensorflow as tf

import cv2
from collections import defaultdict
import time
import os
import random
random.seed(5487)

from model import Model
from utils import document, plot_doc, write_info, write_test,visualize

############################
#         Parameters       #
############################
H_3 = 24
W_3 = 48

############################
#           Flags          #
############################
# --------- Mode --------- #
tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_boolean("restore", False, "training mode.")
# ---- Hyperparameters --- #
tf.app.flags.DEFINE_float("learning_rate", 0.005, "Number of labels.")
tf.app.flags.DEFINE_integer("epoch", 100, "Number of epoch.")
tf.app.flags.DEFINE_integer("batch", 3, "Number of batches(20*videos).")
tf.app.flags.DEFINE_integer("slice", 10, "Number of videos.") # small batch
# ------ Save Model ------ #
tf.app.flags.DEFINE_string("data_dir", "../Tool/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Weights directory")
tf.app.flags.DEFINE_string("test_dir", "3", "Test directory")
tf.app.flags.DEFINE_integer("per_checkpoint", 2000, "How many steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("inference_version", -1, "The version for inferencing.")
tf.app.flags.DEFINE_integer("val_num", 10, "number of validation images.")

FLAGS = tf.app.flags.FLAGS

############################
#         Functions        #
############################
def get_data(is_train=FLAGS.is_train):
    left_eye = [] # (4-d tensor) shape : n, w, h, 3
    right_eye = [] # (4-d tensor) shape : n, w, h, 3
    gt = [] # shape : n
    
    if is_train:
        for i in range(FLAGS.batch):
            print("Getting the "+str(i)+"-th batch.")
            data_ = np.load(FLAGS.data_dir+str(i)+'/second_data.npy', allow_pickle=True)
            left_eye_, right_eye_, gt_ = data_[0], data_[1], data_[2]
            # shuffle
            shuffle_index = np.arange(left_eye_.shape[0])
            np.random.shuffle(shuffle_index)
            # append
            for j in range(left_eye_.shape[0]):
                idx = shuffle_index[j]
                left_eye.append(left_eye_[idx])
                right_eye.append(right_eye_[idx])
                gt.append(gt_[idx])

    else:
        print("Getting the test data.")
        data_ = np.load(FLAGS.data_dir+FLAGS.test_dir+'/second_data.npy', allow_pickle=True)
        left_eye_, right_eye_, gt_ = data_[0], data_[1], data_[2]
        for i in range(len(left_eye_)):
            left_eye.append(left_eye_[i])
            right_eye.append(right_eye_[i])
            gt.append(gt_[i])

    return left_eye, right_eye, gt

def train(model, sess, left_eye, right_eye, gt): 
    loss, acc = 0.0, 0.0
    st, ed, times = 0, 0, 0
    max_len = len(left_eye) # shape : [n, 2, H, W]

    # for every slice
    while st < max_len:
        ed = st + FLAGS.slice if st + FLAGS.slice < max_len else max_len
        print("Trainig 1 slice, from "+str(st)+" to "+str(ed-1)+", ...")
        feed = {model.left: left_eye[st:ed], model.right: right_eye[st:ed], model.y: gt[st:ed]}
        loss_, acc_, _ = sess.run([model.loss, model.acc, model.train_op], feed_dict=feed)
        loss += loss_
        acc += acc_
        times += 1
        st = ed
    
    loss/=times
    acc/=times

    return loss, acc

def test(model, sess, left_eye, right_eye, gt):
    feed = {model.left: left_eye, model.right: right_eye, model.y: gt}
    loss, acc = sess.run([model.loss, model.acc], feed_dict=feed)
    
    return loss, acc

############################
#           Main           #
############################

# ------- variables ------ #
best_acc = 0
best_epoch = 0

# -------- config -------- #
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# ------- run sess ------- #
with tf.Session(config=config) as sess:
    if FLAGS.is_train:
        
        # building computational graph (model)
        if FLAGS.restore:
            model = Model(learning_rate=FLAGS.learning_rate)
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)
            model.saver.restore(sess, model_path)
        else:    
            model = Model(learning_rate=FLAGS.learning_rate)
            tf.global_variables_initializer().run()
        
        # train list (for documenting loss and acc.)
        doc = defaultdict(list)
        
        # training data
        left_eye, right_eye, gt = get_data()

        # get validation data
        left_eye_val, right_eye_val, gt_val = left_eye[:FLAGS.val_num], right_eye[:FLAGS.val_num], gt[:FLAGS.val_num]
        left_eye, right_eye, gt = left_eye[FLAGS.val_num:], right_eye[FLAGS.val_num:], gt[FLAGS.val_num:]

        for epoch in range(FLAGS.epoch):
            start_time = time.time()
            train_loss, train_acc = train(model, sess, left_eye, right_eye, gt)
            val_loss, val_acc = test(model, sess, left_eye_val, right_eye_val, gt_val)
            
            acc = val_acc
            if acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch + 1
                model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, global_step=model.global_step)

            epoch_time = time.time() - start_time
            document(epoch, epoch_time, train_loss, train_acc, val_loss, val_acc, doc)
        
        plot_doc(doc)
        write_info(best_epoch, best_acc)

    else:
        model = Model()
        if FLAGS.inference_version == -1:
            print("Please set the inference version!")
        else:
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)

        model.saver.restore(sess, model_path)
        left_eye, right_eye, gt = get_data()
        test_acc = test(model, sess, left_eye, right_eye, gt)
        write_test(test_acc)
