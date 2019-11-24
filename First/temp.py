
def gradient_generator(G1, G2, G3):
    # G: [height, width, 3]
    height = G2.shape[0]
    width = G2.shape[1]

    candidate_list = []
    for i in range(2, height-2):
        for j in range(2, width-2):
            G1_max = np.max(G1[i-2:i+3, j-2:j+3])
            G2_max = np.max(G2[i-2:i+3, j-2:j+3])
            G3_max = np.max(G3[i-2:i+3, j-2:j+3])
            G1_min = np.min(G1[i-2:i+3, j-2:j+3])
            G2_min = np.min(G2[i-2:i+3, j-2:j+3])
            G3_min = np.min(G3[i-2:i+3, j-2:j+3])
            maximum = np.max([G1_max, G2_max, G3_max])
            minimum = np.min([G1_min, G2_min, G3_min])
            if np.max(G2[i][j]) == maximum or np.min(G2[i][j]) == minimum:
                Gr = (sum(pow(pow(G2[i+1][j]-G2[i-1][j], 2)+pow(G2[i][j+1]-G2[i-1][j-1], 2), 0.5)), i, j)
                candidate_list.append(Gr)
                
    candidate_list = sorted(candidate_list, key = lambda x : x[0], reverse=True)
    return candidate_list

    def get_candidate_regions(self, region_points, partial):
        for i in range(self.frames):
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
            #candidate_point = []
            min_dis_left = THRESHOLD_LEFT
            min_dis_right = THRESHOLD_RIGHT
            min_left_idx = -1
            min_right_idx = -1

            for j in range(min_num):
                # if we are confident about that this point is the eye center, append it to the list.
                left_dis = pow(candidate_list[j][1]-left_center_x, 2)+pow(candidate_list[j][2]-left_center_y, 2)
                right_dis = pow(candidate_list[j][1]-right_center_x, 2)+pow(candidate_list[j][2]-right_center_y, 2)
                
                # position of the candidate point
                if left_dis < min_dis_left:
                    min_dis_left = left_dis
                    min_left_idx = j
                if right_dis < min_dis_right:
                    min_dis_right = right_dis
                    min_right_idx = j
            
            if (partial):# shape : [n (0 or 1 or 2), 4 (2, 1, 1, 1)] -> n points
                if(min_left_idx != -1):
                    region_points.append([[candidate_list[min_left_idx][1], candidate_list[min_left_idx][2]], 0, self.video_idx, i])
                if(min_right_idx != -1):
                    region_points.append([[candidate_list[min_right_idx][1], candidate_list[min_right_idx][2]], 1, self.video_idx, i])
            else :# shape : [frames, 2, 2]
                if(min_left_idx != -1 and min_right_idx != -1):
                    region_points.append([[candidate_list[min_left_idx][1], candidate_list[min_left_idx][2]], [candidate_list[min_right_idx][1], candidate_list[min_right_idx][2]]])
                elif(min_left_idx != -1 and min_right_idx == -1):
                    region_points.append([[candidate_list[min_left_idx][1], candidate_list[min_left_idx][2]], []])
                elif(min_left_idx == -1 and min_right_idx != -1):
                    region_points.append([[], [candidate_list[min_right_idx][1], candidate_list[min_right_idx][2]]])
                else:
                    region_points.append([[], []])



'''
# TODO:
- try gray-scale images.
- shuffle regions in train() function.
'''