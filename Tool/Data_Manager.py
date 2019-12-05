import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

############################
#        Data Manager      #
############################
class Data_Manager:
    def __init__(self, path, W_1=24, H_1=12, W_2=36, H_2=18, W_3=48, H_3=24, width=180, height=360):
        self.path = path
        # kernel size
        self.W_1 = W_1
        self.H_1 = H_1
        self.W_2 = W_2
        self.H_2 = H_2
        self.W_3 = W_3
        self.H_3 = H_3
        self.width = width
        self.height = height

    # ------ To candidate list ------ #
    # ENTRY
    def to_candidate_list(self, kernel_SD=[0.0, 0.5, 0.9], num_candidates=200):
        # Get data
        videos = np.load(self.path+'/video.npy', allow_pickle=True)
        region_points = np.load(self.path+'/region_point.npy')
        candidate_list = []

        # process the data for training...
        for i in range(len(videos)):
            Gaussian_video = []
            for img in videos[i]:
                blur = [None, None, None]
                blur[0] = cv2.Sobel(cv2.cvtColor(cv2.GaussianBlur(img, (3,3), kernel_SD[0]), cv2.COLOR_BGR2GRAY),cv2.CV_64F,0,1,ksize=3)
                blur[1] = cv2.Sobel(cv2.cvtColor(cv2.GaussianBlur(img, (3,3), kernel_SD[1]), cv2.COLOR_BGR2GRAY),cv2.CV_64F,0,1,ksize=3)
                blur[2] = cv2.Sobel(cv2.cvtColor(cv2.GaussianBlur(img, (3,3), kernel_SD[2]), cv2.COLOR_BGR2GRAY),cv2.CV_64F,0,1,ksize=3)
                Gaussian_video.append(blur)
            
            for j in range(len(videos[i])):
                #if j%2 == 1:
                #    continue
                print("Processing the "+str(j)+"-th frame...")
                G1 = Gaussian_video[j][0] # the first imge in the j-th frame
                G2 = Gaussian_video[j][1] # the second imge in the j-th frame
                G3 = Gaussian_video[j][2] # the third imge in the j-th frame
                candidate_list_ = self.gradient_generator(G1, G2, G3, num_candidates)
                candidate_list.append(candidate_list_)

        np.save(self.path+"/candidate_list", candidate_list)

    # ------ To First data ------ #
    # ENTRY
    def to_First_data(self, threshold_left=150, threshold_right=150, HSV_convert=[[0.005, -0.46, -0.54],[0.03, 0.09, 0.45],[0.01, 0.27, -0.38],[-0.04, -0.61, 0.32],[-0.01, 0.6, 0.2]]):
        videos = np.load(self.path+'/video.npy', allow_pickle=True)
        candidate_list = np.load(self.path+'/candidate_list.npy', allow_pickle=True)
        region_points = np.load(self.path+'/region_point.npy')
        data_1 = []
        data_2 = []
        data_3 = []
        gt = []

        all_=0
        have=0
        
        for i in range(len(videos)):
            # get the center point of the eye.
            left_center_X = region_points[i][0][0]
            left_center_Y = region_points[i][0][1]
            right_center_X = region_points[i][1][0]
            right_center_Y = region_points[i][1][1]

            for j in range(len(videos[i])):
                #if j%2 == 1:
                #    continue
                print("Processing the "+str(j)+"-th frame...")
                get=False
                # for storing the candidate points...
                zero_label_list = random.sample(range(0,len(candidate_list[i])),5)
                for k in range(len(candidate_list[i])):
                    X = candidate_list[i][k][2]
                    Y = candidate_list[i][k][1]
                    in_1, in_2, in_3 = self.cut_region([Y, X], videos[i][j])
                    
                    assert in_1.shape == (self.H_1, self.W_1, 3)
                    assert in_2.shape == (self.H_2, self.W_2, 3)
                    assert in_3.shape == (self.H_3, self.W_3, 3)

                    # if we are confident about that this point is the eye center, append it to the list.
                    left_dis = pow(X-left_center_X, 2)+pow(Y-left_center_Y, 2)
                    right_dis = pow(X-right_center_X, 2)+pow(Y-right_center_Y, 2)
                    
                    # position of the candidate point
                    if left_dis < threshold_left:
                        gt.append(1)
                        data_1.append(in_1)
                        data_2.append(in_2)
                        data_3.append(in_3)

                        for t in range(5):
                            data_1.append(self.dye_image(in_1, HSV_convert[t]))
                            data_2.append(self.dye_image(in_2, HSV_convert[t]))
                            data_3.append(self.dye_image(in_3, HSV_convert[t]))
                            gt.append(1)
                        #print("frame ["+str(j)+"] : left!")
                        #self.show_img(in_3, 1)
                        get=True

                    elif right_dis < threshold_right:
                        gt.append(2)
                        data_1.append(in_1)
                        data_2.append(in_2)
                        data_3.append(in_3)
                        for t in range(5):
                            data_1.append(self.dye_image(in_1, HSV_convert[t]))
                            data_2.append(self.dye_image(in_2, HSV_convert[t]))
                            data_3.append(self.dye_image(in_3, HSV_convert[t]))
                            gt.append(1)
                        #print("frame ["+str(j)+"] : right!")
                        #self.show_img(in_3, 2)
                        get=True

                    else:
                        if k in zero_label_list:
                            gt.append(0)
                            data_1.append(in_1)
                            data_2.append(in_2)
                            data_3.append(in_3)
                            for t in range(5):
                                data_1.append(self.dye_image(in_1, HSV_convert[t]))
                                data_2.append(self.dye_image(in_2, HSV_convert[t]))
                                data_3.append(self.dye_image(in_3, HSV_convert[t]))
                                gt.append(1)

                #self.visualize(videos[i][j],candidate_list[i])
                all_ += 1
                have = have + 1 if get else have

        print(have/all_)
        print(len(data_1))
        np.save(self.path+"/first_data", [data_1, data_2, data_3, gt])

    def cut_region(self, center_in, frame):
        in_Y_cut, in_X_cut = center_in[0], center_in[1]
        padding_frame = np.zeros(shape=(self.H_3*2+self.height, self.W_3*2+self.width, 3), dtype=int)
        padding_frame[self.H_3:self.H_3+self.height, self.W_3:self.W_3+self.width] += frame
        
        if(in_Y_cut < 0 or in_X_cut < 0):
            return padding_frame[0:self.H_1, 0:self.W_1], padding_frame[0:self.H_2, 0:self.W_2], padding_frame[0:self.H_3, 0:self.W_3]

        st_Y_1 = (int)(in_Y_cut-self.H_1/2+self.H_3+self.H_1/6)
        ed_Y_1 = (int)(in_Y_cut+self.H_1/2+self.H_3+self.H_1/6)
        st_Y_2 = (int)(in_Y_cut-self.H_2/2+self.H_3+self.H_2/6)
        ed_Y_2 = (int)(in_Y_cut+self.H_2/2+self.H_3+self.H_2/6)
        st_Y_3 = (int)(in_Y_cut-self.H_3/2+self.H_3+self.H_3/6)
        ed_Y_3 = (int)(in_Y_cut+self.H_3/2+self.H_3+self.H_3/6)

        st_X_1 = (int)(in_X_cut-self.W_1/2)+self.W_3
        ed_X_1 = (int)(in_X_cut+self.W_1/2)+self.W_3
        st_X_2 = (int)(in_X_cut-self.W_2/2)+self.W_3
        ed_X_2 = (int)(in_X_cut+self.W_2/2)+self.W_3
        st_X_3 = (int)(in_X_cut-self.W_3/2)+self.W_3
        ed_X_3 = (int)(in_X_cut+self.W_3/2)+self.W_3
        
        return padding_frame[st_Y_1:ed_Y_1, st_X_1:ed_X_1], padding_frame[st_Y_2:ed_Y_2, st_X_2:ed_X_2], padding_frame[st_Y_3:ed_Y_3, st_X_3:ed_X_3]

    def visualize(self, frame, candidate_list):
        fig,ax = plt.subplots(1)
        ax.imshow(frame)
        print(len(candidate_list))
        for i in range(len(candidate_list)):
            plt.scatter(candidate_list[i][2], candidate_list[i][1], color='red', linewidths=0.01)
        plt.show()

    def gradient_generator(self, G1, G2, G3, num_candidates):
        candidate_list = []
        for i in range(2, self.height-2):
            if (i%2 == 1):
                continue
            for j in range(2, self.width-2):
                if (j%2 == 1):
                    continue
                G1_max = np.max(G1[i-2:i+3, j-2:j+3])
                G2_max = np.max(G2[i-2:i+3, j-2:j+3])
                G3_max = np.max(G3[i-2:i+3, j-2:j+3])
                maximum = np.max([G1_max, G2_max, G3_max])
                if np.max(G2[i][j]) == maximum:
                    candidate_list.append([G2[i][j], i, j])
                    
        candidate_list = sorted(candidate_list, key = lambda x : x[0], reverse=True)
        candidate_list_ = candidate_list[0:num_candidates]
        return candidate_list_

    # ------ To cut Regions ------ #
    # ENTRY
    def to_Second_data(self, random_show=False, HSV_convert=[[0.005, -0.46, -0.54],[0.03, 0.09, 0.45],[0.01, 0.27, -0.38],[-0.04, -0.61, 0.32],[-0.01, 0.6, 0.2]]):
        
        videos = np.load(self.path+'/video.npy', allow_pickle=True)
        region_points = np.load(self.path+'/region_point.npy')
        ground_truth = np.load(self.path+'/ground_truth.npy')
        left_eyes, right_eyes, gt = [], [], []
        print(region_points)

        for i in range(len(videos)):
            
            left_X = region_points[i][0][0]
            left_Y = region_points[i][0][1]
            right_X = region_points[i][1][0]
            right_Y = region_points[i][1][1]

            if random_show :
                fi = random.randrange(len(videos[i]))
                _, _, left = self.cut_region([left_Y, left_X], videos[i][fi])
                _, _, right = self.cut_region([right_Y, right_X], videos[i][fi])
                self.show_img(left, ground_truth[i])
                self.show_img(right, ground_truth[i])
            
            for j in range(len(videos[i])):
                print("Processing the "+str(j)+"-th frame...")
                _, _, left = self.cut_region([left_Y, left_X], videos[i][j])
                _, _, right = self.cut_region([right_Y, right_X], videos[i][j])
                left_eyes.append(left)
                right_eyes.append(right)
                gt.append(ground_truth[i])
                for k in range(5):
                    dye_img_left = self.dye_image(left, HSV_convert[k])
                    dye_img_right = self.dye_image(right, HSV_convert[k])
                    left_eyes.append(dye_img_left)
                    right_eyes.append(dye_img_right)
                    gt.append(ground_truth[i])
        
        assert len(left_eyes) == len(right_eyes)
        assert len(left_eyes) == len(gt)

        np.save(self.path+"/second_data", [left_eyes, right_eyes, gt])
    
    def dye_image(self, img, HSV_convert):
        img_HSV = cv2.cvtColor(img.astype(np.float32)/255.0, cv2.COLOR_RGB2HSV)
        img_HSV[:, :, 0] = img_HSV[:, :, 0]+img_HSV[:, :, 0]*HSV_convert[0]
        img_HSV[:, :, 1] = img_HSV[:, :, 1]+img_HSV[:, :, 1]*HSV_convert[1]
        img_HSV[:, :, 2] = img_HSV[:, :, 2]+img_HSV[:, :, 2]*HSV_convert[2]
        dye_img = (cv2.cvtColor(img_HSV, cv2.COLOR_HSV2RGB)*255).astype(np.int)
        return dye_img

    def show_img(self, img, gt):
        fig,ax = plt.subplots(1)
        ax.imshow(img)
        plt.title(str(gt))
        plt.show()

    # ------ To cut Regions ------ #
    # ENTRY
    # to second plus
    def to_Second_plus(self, threshold_left=150, threshold_right=150, HSV_convert=[[0.005, -0.46, -0.54],[0.03, 0.09, 0.45],[0.01, 0.27, -0.38],[-0.04, -0.61, 0.32],[-0.01, 0.6, 0.2]]):
        videos = np.load(self.path+'/video.npy', allow_pickle=True)
        candidate_list = np.load(self.path+'/candidate_list.npy', allow_pickle=True)
        region_points = np.load(self.path+'/region_point.npy')
        ground_truth = np.load(self.path+'/ground_truth.npy')

        left_eyes, right_eyes, gt = [], [], []

        all_=0
        have=0
        
        for i in range(len(videos)):
            # get the center point of the eye.
            left_center_X = region_points[i][0][0]
            left_center_Y = region_points[i][0][1]
            right_center_X = region_points[i][1][0]
            right_center_Y = region_points[i][1][1]

            for j in range(len(videos[i])):
                print("Processing the "+str(j)+"-th frame...")
                get=False
                best_left_dis = threshold_left
                best_right_dis = threshold_right
                best_left = [np.zeros(shape=(self.H_1, self.W_1, 3), dtype=int), np.zeros(shape=(self.H_2, self.W_2, 3), dtype=int), np.zeros(shape=(self.H_3, self.W_3, 3), dtype=int)]
                best_right = [np.zeros(shape=(self.H_1, self.W_1, 3), dtype=int), np.zeros(shape=(self.H_2, self.W_2, 3), dtype=int), np.zeros(shape=(self.H_3, self.W_3, 3), dtype=int)]
                # for storing the candidate points...
                for k in range(len(candidate_list[i])):
                    X = candidate_list[i][k][2]
                    Y = candidate_list[i][k][1]

                    # if we are confident about that this point is the eye center, append it to the list.
                    left_dis = pow(X-left_center_X, 2)+pow(Y-left_center_Y, 2)
                    right_dis = pow(X-right_center_X, 2)+pow(Y-right_center_Y, 2)
                    
                    # position of the candidate point
                    if left_dis < best_left_dis:
                        left = [None, None, None]
                        left[0], left[1], left[2] = self.cut_region([Y, X], videos[i][j])
                        best_left_dis = left_dis
                        best_left = left
                        get=True

                    elif right_dis < best_right_dis:
                        right = [None, None, None]
                        right[0], right[1], right[2] = self.cut_region([Y, X], videos[i][j])
                        best_right_dis = right_dis
                        best_right = right
                        get=True

                all_ += 1
                have = have + 1 if get else have

                for k in range(3):
                    left_img = cv2.resize(best_left[k], dsize=(self.W_3, self.H_3), interpolation=cv2.INTER_NEAREST)
                    right_img = cv2.resize(best_right[k], dsize=(self.W_3, self.H_3), interpolation=cv2.INTER_NEAREST)
                    left_eyes.append(left_img)
                    right_eyes.append(right_img)
                    gt.append(ground_truth[i])
                    for l in range(5):
                        dye_img_left = self.dye_image(left_img, HSV_convert[l])
                        dye_img_right = self.dye_image(right_img, HSV_convert[l])
                        left_eyes.append(dye_img_left)
                        right_eyes.append(dye_img_right)
                        gt.append(ground_truth[i])

        print(have/all_)
        assert left_eyes[0].shape == right_eyes[0].shape
        assert len(left_eyes) == len(gt)
        assert len(right_eyes) == len(gt)
        np.save(self.path+"/second_data_p", [left_eyes, right_eyes, gt])