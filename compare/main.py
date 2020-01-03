import tensorflow as tf
import numpy as np
import cv2
from model_first import Model as model_f
#from model_second import Model as model_s
from utils import get_eye, get_candidate_list, get_First_data #get_label

W_3 = 48
H_3 = 24

############################
#           Flags          #
############################
tf.app.flags.DEFINE_boolean("store_data", False, "set True to store the outputs of the first layer.")
FLAGS = tf.app.flags.FLAGS

############################
#           Main           #
############################
config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    
    first_model = model_f()
    first_model.saver.restore(sess, 'checkpoint-00200037')
    #second_model = model_s()
    #second_model.saver.restore(sess, 'second')

    left_eye_list = []
    right_eye_list = []
    last_left_eye = np.zeros(shape=(H_3, W_3, 3), dtype=int)
    last_right_eye = np.zeros(shape=(H_3, W_3, 3), dtype=int)
    all_left = 0
    all_right = 0
    correct_left = 0
    correct_right = 0

    videos = np.load('../dataset/test_unknown/video.npy', allow_pickle=True)
    region_points = np.load('../dataset/test_unknown/region_point.npy')
    for i in range(len(videos)):
        print("processing "+str(i)+"-th video...")
        rp = region_points[i]
        for frame in videos[i]:
            print("processing one frame...")

            have_left = False
            have_right = False

            data_1, data_2, data_3, candidate_list = get_First_data(frame)
            left_eye, right_eye, left_region, right_region = get_eye(first_model, sess, data_1, data_2, data_3, candidate_list, last_left_eye, last_right_eye)
            #print(region_points[0], left_region, right_region)
            #label = get_label(model, sess, left_eye, right_eye)
            #print(label)
            #last_left_eye = left_eye
            #last_right_eye = right_eye
            
            if left_region:
                left_dis = min(pow(left_region[0]-rp[0][0], 2)+pow(left_region[1]-rp[0][1], 2), pow(left_region[0]-rp[1][0], 2)+pow(left_region[1]-rp[1][1], 2))
                if left_dis<150:
                    have_left = True
                cv2.rectangle(frame,(left_region[0]-24,left_region[1]-24),(left_region[0]+24,left_region[1]+24),( 0 , 255 , 0 ), 2 )
                print("LEFT:", have_left)
                

            if right_region:
                right_dis = min(pow(right_region[0]-rp[0][0], 2)+pow(right_region[1]-rp[0][1], 2), pow(right_region[0]-rp[1][0], 2)+pow(right_region[1]-rp[1][1], 2))
                if right_dis<150:
                    have_right = True
                cv2.rectangle(frame,(right_region[0]-24,right_region[1]-24),(right_region[0]+24,right_region[1]+24),( 0 , 255 , 0 ), 2 )
                print("RIGHT", have_right)
                    
            if rp[0][0]!=-1:
                all_left = all_left+1
                correct_left = correct_left+1 if have_left else correct_left

            if rp[1][0]!=-1:
                all_right = all_right+1
                correct_right = correct_right+1 if have_right else correct_right

            b,g,r = cv2.split(frame)  
            img2 = cv2.merge([r,g,b])
            cv2.imshow("img", img2)  
            cv2.waitKey( 0 )
            break
    
    #print(all_left, all_right, correct_left, correct_right)