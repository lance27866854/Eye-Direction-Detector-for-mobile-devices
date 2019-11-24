import cv2
import numpy as np
from matplotlib import pyplot as plt

############################
#         Parameters       #
############################
THRESHOLD_LEFT = 800
THRESHOLD_RIGHT = 800

############################
#         Functions        #
############################
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

############################
#           Videos         #
############################
class Video:
    def __init__(self, video, label_region, video_idx, kernel_SD=[0.3, 0.5, 0.9], num_candidates=150):
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
        self.video = video
        self.video_idx = video_idx

    def visualize(self, candidate_list, idx):
        fig,ax = plt.subplots(1)
        ax.imshow(self.video[0])
        min_num = len(candidate_list) if len(candidate_list) < self.num_candidates else self.num_candidates
        for i in range(min_num):
            plt.scatter(candidate_list[i][0][0][0], candidate_list[i][0][0][1], color='cornflowerblue', linewidths=0.5)
        #plt.savefig('img/'+str(idx)+'.png')
        plt.show()

    def get_candidate_regions(self, region_cet):# shape : [n, 4(2, 1, 1, 1)]
        for i in range(self.frames):
            print("Processing the "+str(i)+"-th frame...")
            G1 = self.Gaussian_video[i][0] # the first imge in the i-th frame
            G2 = self.Gaussian_video[i][1] # the second imge in the i-th frame
            G3 = self.Gaussian_video[i][2] # the third imge in the i-th frame
            candidate_list = gradient_generator(G1, G2, G3)
            min_num = len(candidate_list) if len(candidate_list) < self.num_candidates else self.num_candidates
            
            # get the center point of the eye.
            left_center_x = self.label_region[0][0]+self.label_region[0][2]/2
            left_center_y = self.label_region[0][1]+self.label_region[0][3]/2
            right_center_x = self.label_region[1][0]+self.label_region[1][2]/2
            right_center_y = self.label_region[1][1]+self.label_region[1][3]/2

            # for storing the candidate points...
            for j in range(min_num):
                # if we are confident about that this point is the eye center, append it to the list.
                left_dis = pow(candidate_list[j][1]-left_center_x, 2)+pow(candidate_list[j][2]-left_center_y, 2)
                right_dis = pow(candidate_list[j][1]-right_center_x, 2)+pow(candidate_list[j][2]-right_center_y, 2)
                
                # position of the candidate point
                if left_dis < THRESHOLD_LEFT:
                    region_cet.append([[candidate_list[j][1], candidate_list[j][2]], 1, self.video_idx, i])
                elif right_dis < THRESHOLD_RIGHT:
                    region_cet.append([[candidate_list[j][1], candidate_list[j][2]], 2, self.video_idx, i])
                else:
                    region_cet.append([[candidate_list[j][1], candidate_list[j][2]], 0, self.video_idx, i])
    
    def get_raw_candidate_regions(self):
        frame_region = []# shape : [frame, n, 2]
        for i in range(self.frames):
            print("Processing the "+str(i)+"-th frame...")
            G1 = self.Gaussian_video[i][0] # the first imge in the i-th frame
            G2 = self.Gaussian_video[i][1] # the second imge in the i-th frame
            G3 = self.Gaussian_video[i][2] # the third imge in the i-th frame
            candidate_list = gradient_generator(G1, G2, G3)
            min_num = len(candidate_list) if len(candidate_list) < self.num_candidates else self.num_candidates
            n_points = []
            # for storing the candidate points...
            for j in range(min_num):
                n_points.append([candidate_list[j][1], candidate_list[j][2]])
            frame_region.append(n_points)

        return frame_region