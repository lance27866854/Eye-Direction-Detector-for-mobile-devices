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
NAME = '6'
FILE_NUM = 0
ANSWER = []
REGION_POINTS = []

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
data_manager = Data_Manager(path='../dataset/test_known')
data_manager.to_First_data(zero_label=5, data_augmentation=True,HSV_convert=[[0.005, -0.46, -0.54],[0.03, 0.09, 0.45],[0.01, 0.27, -0.38]],angle_convert=[-5,0,5])
data_manager = Data_Manager(path='../dataset/test_unknown')
data_manager.to_First_data(zero_label=5, data_augmentation=True,HSV_convert=[[0.005, -0.46, -0.54],[0.03, 0.09, 0.45],[0.01, 0.27, -0.38]],angle_convert=[-5,0,5])

# (Second) Convert the data to .npy
for i in range(8):
        data_manager = Data_Manager(path='../dataset/'+str(i))
        data_manager.to_Second_data(data_augmentation=True)
        data_manager.to_Second_data(data_augmentation=False)

