import cv2
import numpy as np
############################
#           First          #
############################

def cut_region(center_in, frame, W_1=30, H_1=20, W_2=42, H_2=28, W_3=54, H_3=36, width=180, height=360):
    in_Y_cut, in_X_cut = center_in[0], center_in[1]
    padding_frame = np.zeros(shape=(H_3*2+height, W_3*2+width, 3), dtype=float)
    padding_frame[H_3:H_3+height, W_3:W_3+width] += frame
    
    st_Y_1 = (int)(in_Y_cut-H_1/2+H_3)
    ed_Y_1 = (int)(in_Y_cut+H_1/2+H_3)
    st_Y_2 = (int)(in_Y_cut-H_2/2+H_3)
    ed_Y_2 = (int)(in_Y_cut+H_2/2+H_3)
    st_Y_3 = (int)(in_Y_cut-H_3/2+H_3)
    ed_Y_3 = (int)(in_Y_cut+H_3/2+H_3)
    st_X_1 = (int)(in_X_cut-W_1/2)+W_3
    ed_X_1 = (int)(in_X_cut+W_1/2)+W_3
    st_X_2 = (int)(in_X_cut-W_2/2)+W_3
    ed_X_2 = (int)(in_X_cut+W_2/2)+W_3
    st_X_3 = (int)(in_X_cut-W_3/2)+W_3
    ed_X_3 = (int)(in_X_cut+W_3/2)+W_3
    
    return padding_frame[st_Y_1:ed_Y_1, st_X_1:ed_X_1], padding_frame[st_Y_2:ed_Y_2, st_X_2:ed_X_2], padding_frame[st_Y_3:ed_Y_3, st_X_3:ed_X_3]

def get_candidate_list(frame, kernel_SD=[0.0, 0.5, 0.9], num_candidates=200, width=180, height=360):
    candidate_list = []
    blur = [None, None, None]
    blur[0] = cv2.Sobel(cv2.cvtColor(cv2.GaussianBlur(frame, (3,3), kernel_SD[0]), cv2.COLOR_BGR2GRAY),cv2.CV_64F,0,1,ksize=3)
    blur[1] = cv2.Sobel(cv2.cvtColor(cv2.GaussianBlur(frame, (3,3), kernel_SD[1]), cv2.COLOR_BGR2GRAY),cv2.CV_64F,0,1,ksize=3)
    blur[2] = cv2.Sobel(cv2.cvtColor(cv2.GaussianBlur(frame, (3,3), kernel_SD[2]), cv2.COLOR_BGR2GRAY),cv2.CV_64F,0,1,ksize=3)
    
    for i in range(2, height-2):
        if (i%2 == 1):
            continue
        for j in range(2, width-2):
            if (j%2 == 1):
                continue
            G1_max = np.max(blur[0][i-2:i+3, j-2:j+3])
            G2_max = np.max(blur[1][i-2:i+3, j-2:j+3])
            G3_max = np.max(blur[2][i-2:i+3, j-2:j+3])
            maximum = np.max([G1_max, G2_max, G3_max])
            if np.max(blur[1][i][j]) == maximum:
                candidate_list.append([blur[1][i][j], i, j])
                    
    candidate_list = sorted(candidate_list, key = lambda x : x[0], reverse=True)
    candidate_list_ = candidate_list[0:num_candidates]
    return candidate_list_

def get_First_data(frame, threshold_left=150, threshold_right=150):
    data_1 = []
    data_2 = []
    data_3 = []

    candidate_list = get_candidate_list(frame)
    for k in range(len(candidate_list)):
        in_1, in_2, in_3 = cut_region([candidate_list[k][1], candidate_list[k][2]], frame)
        data_1.append(in_1)
        data_2.append(in_2)
        data_3.append(in_3)

    return data_1, data_2, data_3, candidate_list
def get_eye(model, sess, data_1, data_2, data_3, candidate_list, left_default, right_default):
    
    left_max_logit = 0
    right_max_logit = 0
    left_max_img = left_default
    right_max_img = right_default
    
    left_region = None
    right_region = None
    left_window_idx = 0
    right_window_idx = 0

    length = len(candidate_list)
    for i in range(length):
        #print("processing the "+str(i+1)+"-th image.")
        
        pred = [0, 0, 0]
        logit = [0, 0, 0]
        img = [data_1[i], data_2[i], data_3[i]]

        feed = {model.x_1: [img[0]], model.x_2: [img[1]], model.x_3: [img[2]]}
        pred[0], pred[1], pred[2], logit[0], logit[1], logit[2] = sess.run([model.val_pred_1, model.val_pred_2, model.val_pred_3, model.val_logit_1, model.val_logit_2, model.val_logit_3], feed)
        max_logit = np.argmax(np.array([max(logit[0][0]), max(logit[1][0]), max(logit[2][0])]))
        logit_value = logit[max_logit][0]
        
        if pred[max_logit] == 1 and logit_value[max_logit] > left_max_logit: # left
            left_max_logit = logit_value[max_logit]
            left_max_img = img[max_logit]
            left_region = [candidate_list[i][2], candidate_list[i][1]]
            left_window_idx = max_logit

        if pred[max_logit] == 2 and logit_value[max_logit] > right_max_logit: # right
            right_max_logit = logit_value[max_logit]
            right_max_img = img[max_logit]
            right_region = [candidate_list[i][2], candidate_list[i][1]]
            right_window_idx = max_logit

    return left_window_idx, right_window_idx, left_region, right_region #left_max_img, right_max_img


############################
#          Second          #
############################
def get_second_eye(eye, frame, last_left_eye, last_right_eye):
    left_window_idx = eye[0]
    right_window_idx = eye[1]
    left_region = eye[2]
    right_region = eye[3]
    
    left_prev = False
    right_prev = False
    
    left_eye = [None,None,None]
    right_eye = [None,None,None]
    if left_region:
        left_eye[0], left_eye[1], left_eye[2] = cut_region([left_region[1], left_region[0]], frame)
    else:
        left_prev = True
    
    if right_region:
        right_eye[0], right_eye[1], right_eye[2] = cut_region([right_region[1], right_region[0]], frame)
    else:
        right_prev = True
    
    out_left = cv2.resize(left_eye[left_window_idx], dsize=(48, 24), interpolation=cv2.INTER_NEAREST) if left_prev==False else last_left_eye
    out_right = cv2.resize(right_eye[right_window_idx], dsize=(48, 24), interpolation=cv2.INTER_NEAREST) if right_prev==False else last_right_eye
    return out_left, out_right


def get_label(model, sess, left_eye, right_eye):
    feed = {model.left: left_eye, model.right: right_eye}
    pred = sess.run([model.pred], feed_dict=feed)
    return pred