import cv2
import numpy as np
from matplotlib import pyplot as plt

T = 70

def gradient_generator(G1, G2, G3):
    # G: [height, width, 3]
    height = G2.shape[0]
    width = G2.shape[1]

    candidate_list = []
    for i in range(2, height-2):
        if (i%2 == 1):
            continue
        for j in range(2, width-2):
            if (j%2 == 1):
                continue
            G1_max = np.max(G1[i-2:i+3, j-2:j+3])
            G2_max = np.max(G2[i-2:i+3, j-2:j+3])
            G3_max = np.max(G3[i-2:i+3, j-2:j+3])
            maximum = np.max([G1_max, G2_max, G3_max])
            if np.max(G2[i][j]) == maximum:
                candidate_list.append((G2[i][j], i, j))
                
    candidate_list = sorted(candidate_list, key = lambda x : x[0], reverse=True)
    return candidate_list


class Video:
    def __init__(self, video, label_region, kernel_SD=[0.0, 0.5, 0.9], num_candidates=200):
        # self.Gaussian_video[0][0].shape == (height, width, 3)
        # self.Gaussian_video == [frames, 3, height, width, 3]
        self.label_region = label_region
        Gaussian_video = []
        for img in video:
            blur = [None, None, None]
            blur[0] = cv2.Sobel(cv2.cvtColor(cv2.GaussianBlur(img, (3,3), kernel_SD[0]), cv2.COLOR_BGR2GRAY),cv2.CV_64F,0,1,ksize=3)
            blur[1] = cv2.Sobel(cv2.cvtColor(cv2.GaussianBlur(img, (3,3), kernel_SD[1]), cv2.COLOR_BGR2GRAY),cv2.CV_64F,0,1,ksize=3)
            blur[2] = cv2.Sobel(cv2.cvtColor(cv2.GaussianBlur(img, (3,3), kernel_SD[2]), cv2.COLOR_BGR2GRAY),cv2.CV_64F,0,1,ksize=3)
            Gaussian_video.append(blur)

        self.Gaussian_video = Gaussian_video
        self.frames = len(Gaussian_video)
        self.num_candidates = num_candidates
        #self.video = video

    def visualize(self, candidate_list, idx):
            fig,ax = plt.subplots(1)
            ax.imshow(self.video[0])
            min_num = len(candidate_list) if len(candidate_list) < self.num_candidates else self.num_candidates
            for i in range(min_num):
                plt.scatter(candidate_list[i][2], candidate_list[i][1], color='cornflowerblue', linewidths=0.5)
            plt.savefig('img/'+str(idx)+'.png')

    def get_candidate_regions(self):
        region_points = []
        for i in range(self.frames):
            print(i)
            G1 = self.Gaussian_video[i][0] # the first imge in the i-th frame
            G2 = self.Gaussian_video[i][1] # the second imge in the i-th frame
            G3 = self.Gaussian_video[i][2] # the third imge in the i-th frame
            candidate_list = gradient_generator(G1, G2, G3)
            #print(len(candidate_list))
            #self.visualize(candidate_list, i)
            min_num = len(candidate_list) if len(candidate_list) < self.num_candidates else self.num_candidates

            candidate_point = []
            for j in range(min_num):
                left_center_x = self.label_region[0][0]+self.label_region[0][2]>>1
                left_center_y = self.label_region[0][1]+self.label_region[0][3]>>1
                right_center_x = self.label_region[1][0]+self.label_region[1][2]>>1
                right_center_y = self.label_region[1][1]+self.label_region[1][3]>>1
                left_loss = pow(candidate_list[j][1]-right_center_x, 2)+pow(candidate_list[j][2]-right_center_y, 2)
                right_loss = pow(candidate_list[j][1]-right_center_x, 2)+pow(candidate_list[j][2]-right_center_y, 2)
                if(left_loss < 100 or right_loss < 100): # Threshold...
                    candidate_point.append([candidate_list[j][1], candidate_list[j][2]])
            print(len(candidate_point)) # shape : [num_points, 2] -> 1 frame
            region_points.append(candidate_point)
        return region_points # shape : [frames, num_points, 2] -> n frame

        