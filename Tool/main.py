from Data_Editor import Data_Editor
from Data_Manager import Data_Manager

# 0->no movement, 1->up, 2->right, 3->down, 4->left
NO = 0
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4

# PARAMETERS
PATH = 'C:\\Users\\user\\Desktop\\Model\\Tool\\source\\'
NAME = '../dataset/6'
FILE_NUM = 20
ANSWER = [RIGHT, RIGHT, LEFT, LEFT, UP,
          NO, DOWN, LEFT, NO, DOWN,
          UP, UP, RIGHT, DOWN, LEFT,
          UP, DOWN, NO, NO, NO,
          UP, UP, DOWN, NO, LEFT,
          LEFT, RIGHT, RIGHT, NO, DOWN, 
          NO, RIGHT, RIGHT, UP, LEFT,
          LEFT, UP, DOWN, DOWN, NO,
          LEFT, UP, DOWN, RIGHT, RIGHT,
          NO, DOWN, RIGHT, UP, LEFT]

REGION_POINTS = [[[ 46,  98], [ 92, 102]],
                 [[ -1,  -1], [ 32, 169]],
                 [[ 40,  95], [ 83, 100]],
                 [[144, 155], [ -1,  -1]],
                 [[ 40,  90], [ 84,  96]],#
                 [[ 34, 170], [112, 171]],
                 [[ 38,  92], [ 83,  95]],
                 [[ 54, 157], [120, 158]],
                 [[ 49,  79], [ 90,  82]],
                 [[ 46, 178], [122, 178]],#
                 [[ 40, 119], [110, 125]],
                 [[154, 139], [ -1,  -1]],
                 [[ 37, 107], [110, 110]],
                 [[ 52, 178], [117, 176]],
                 [[ 32, 108], [107, 111]],#
                 [[ 53, 169], [118, 168]],
                 [[ 30, 116], [103, 119]],
                 [[ 53, 169], [119, 169]],
                 [[ 27, 102], [102, 106]],
                 [[ -1,  -1], [ 19, 186]],#
                 [[ 37, 111], [140, 117]],
                 [[ -1,  -1], [ 22, 142]],
                 [[ 66, 199], [131, 198]],
                 [[ 48, 176], [109, 176]],
                 [[ 56, 183], [116, 186]],#
                 [[ -1,  -1], [ 12, 165]],
                 [[ 58, 149], [126, 149]],
                 [[147,  88], [ -1,  -1]],
                 [[ 33, 163], [102, 161]],
                 [[ 54, 176], [118, 179]],#
                 [[ 46, 166], [125, 168]],
                 [[ 42, 165], [109, 168]],
                 [[ 35, 166], [120, 169]],
                 [[ 50, 162], [108, 165]],
                 [[ 42, 150], [117, 154]],#
                 [[151, 130], [ -1,  -1]],
                 [[ 39, 145], [118, 148]],
                 [[ 38, 164], [109, 168]],
                 [[ 46, 164], [126, 168]],
                 [[ 44, 164], [109, 167]],#
                 [[ -1,  -1], [ 15, 159]],
                 [[ 51, 165], [119, 164]],
                 [[ 42, 173], [120, 169]],
                 [[149, 154], [ -1,  -1]],
                 [[ -1,  -1], [ 22, 174]],#
                 [[ 56, 162], [117, 163]],
                 [[ -1,  -1], [ 19, 170]],
                 [[ 46, 161], [110, 165]],
                 [[ 45, 168], [118, 169]],
                 [[ 54, 165], [124, 167]]]#
# Main
# Store the vector of videos
#data_editor = Data_Editor(name=NAME, answer=ANSWER, file_num=FILE_NUM, path=PATH)
#data_editor.convert_videos_to_files()

# Show the eye-region and store it as a vector.
#for i in range(14, FILE_NUM):
#    data_editor.random_show(REGION_POINTS[i], i)
#data_editor.convert_eye_points_to_files(REGION_POINTS)

# ----------------------------------------------- #
# (First) Convert the data to .npy

#data_manager = Data_Manager(path='../dataset/7')
#data_manager.to_candidate_list(num_candidates=250)
#data_manager.to_First_data(zero_label=5, data_augmentation=True)
#data_manager = Data_Manager(path='../dataset/6')
#data_manager.to_candidate_list(num_candidates=250)
#data_manager.to_First_data(zero_label=5, data_augmentation=True)
#data_manager.to_First_data(zero_label=5, data_augmentation=False)
#for i in range(8):
#        data_manager = Data_Manager(path='../dataset/'+str(i))
#        data_manager.to_First_data(zero_label=5, data_augmentation=True)
data_manager = Data_Manager(path='../dataset/test_known')
data_manager.to_First_data(zero_label=5, data_augmentation=True,HSV_convert=[[0.005, -0.46, -0.54],[0.03, 0.09, 0.45],[0.01, 0.27, -0.38]],angle_convert=[-5,0,5])
data_manager = Data_Manager(path='../dataset/test_unknown')
data_manager.to_First_data(zero_label=5, data_augmentation=True,HSV_convert=[[0.005, -0.46, -0.54],[0.03, 0.09, 0.45],[0.01, 0.27, -0.38]],angle_convert=[-5,0,5])
'''
# (Second) Convert the data to .npy
for i in range(8):
        data_manager = Data_Manager(path='../dataset/'+str(i))
        data_manager.to_Second_data(data_augmentation=True)
        data_manager.to_Second_data(data_augmentation=False)
'''
