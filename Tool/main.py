from Data_Editor import Data_Editor
from Data_Manager import Data_Manager

# 0->no movement, 1->up, 2->right, 3->down, 4->left
NO = 0
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4

# PARAMETERS
PATH = 'C:\\Users\\user\\Desktop\\Model\\Tool\\sources\\'
NAME = '0'
FILE_NUM = 2
ANSWER = [RIGHT, NO]
REGION_POINTS = [[[ 45, 135], [113, 138]],
                 [[ 37, 140], [106, 143]]]

# Main
# Store the vector of videos
data_editor = Data_Editor(name=NAME, answer=ANSWER, file_num=FILE_NUM, path=PATH)
data_editor.convert_videos_to_files()

# Show the eye-region and store it as a vector.
#data_editor.random_show(REGION_POINTS[1], 1)
data_editor.convert_eye_points_to_files(REGION_POINTS)

# ----------------------------------------------- #

# (First) Convert the data to .npy
data_manager = Data_Manager(path=NAME)
#data_manager.to_candidate_list()
data_manager.to_First_data()

# (Second) Convert the data to .npy
#data_manager.to_Second_data(True)
