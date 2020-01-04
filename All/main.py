import tensorflow as tf
import numpy as np
import cv2
from model_first import Model as model_f
from model_second import Model as model_s
from utils import get_eye, get_candidate_list, get_First_data, get_label, get_second_eye
import matplotlib.pyplot as plt

W_3 = 48
H_3 = 24
all_ = 0
have_ = 0

############################
#           Flags          #
############################
tf.app.flags.DEFINE_boolean("first", True, "set True to predict region.")
tf.app.flags.DEFINE_string("test_dir", "test_known", "Test directory")
FLAGS = tf.app.flags.FLAGS

############################
#           Main           #
############################

config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    if FLAGS.first:
        first_model = model_f()
        first_model.saver.restore(sess, './first/first')
    else:
        second_model = model_s()
        second_model.saver.restore(sess, './second/second')

    left_eye_list = []
    right_eye_list = []
    last_left_eye = np.zeros(shape=(H_3, W_3, 3), dtype=int)
    last_right_eye = np.zeros(shape=(H_3, W_3, 3), dtype=int)

    videos = np.load('../dataset/'+FLAGS.test_dir+'/video.npy', allow_pickle=True)
    gt = np.load('../dataset/'+FLAGS.test_dir+'/ground_truth.npy', allow_pickle=True)
    eyes = np.load('eyes/'+FLAGS.test_dir+'.npy', allow_pickle=True)

    if FLAGS.first:
        v_eyes = []
        for i in range(len(videos)):
            eyes = []
            for frame in videos[i]:
                print("processing one frame...")
                data_1, data_2, data_3, candidate_list = get_First_data(frame)
                left_window_idx, right_window_idx, left_region, right_region = get_eye(first_model, sess, data_1, data_2, data_3, candidate_list, last_left_eye, last_right_eye)
                eyes.append([left_window_idx, right_window_idx, left_region, right_region])
            v_eyes.append(eyes)
        np.save('eyes/'+FLAGS.test_dir, v_eyes)
    
    else:
        for i in range(len(videos)):
        last_left_eye = np.zeros(shape=(H_3, W_3, 3), dtype=int)
        last_right_eye = np.zeros(shape=(H_3, W_3, 3), dtype=int)
        for j in range(len(videos[i])):
            print("processing one frame...")
            eye = eyes[i][j]
            left_eye, right_eye = get_second_eye(eye, videos[i][j], last_left_eye, last_right_eye)
            label = get_label(second_model, sess, [left_eye], [right_eye])
            last_left_eye = left_eye
            last_right_eye = right_eye
            
            print(label[0][0], gt[i])
            all_ += 1
            have_ = have_+1 if label[0][0]==gt[i] else have_

        print(have_, all_, have_/all_)