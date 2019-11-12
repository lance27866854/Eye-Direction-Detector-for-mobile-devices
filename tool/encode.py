import cv2
import time
import numpy as np
import string
import os

# PARAMETERS
FILE_NUM = 2
BATCH_NUM = 0
FILE_LOCATION = 'C:\\Users\\user\\Desktop\\tool\\sources\\'
FILE_TYPE = '.mp4'

# tool functions
def write_array(loaders, file_name):
    np.save(file_name, loaders)

def test_coorectly_stored(file_name_info, file_neme):
    file = open(file_name_info, 'r')
    contents = file.readline()
    file.close()
    
    shape_assert = []
    for word in contents.split():
        word = word.strip(string.whitespace)
        shape_assert.append(int(word))
    
    data = np.load(file_neme)
    assert len(data) == shape_assert[0]
    assert data[0][0].shape[0] == shape_assert[1]
    assert data[0][0].shape[1] == shape_assert[2]
    assert data[0][0].shape[2] == shape_assert[3]

# main
# Convert the files (videos) to arrays.
loaders = []
height = None
width = None

for i in range(FILE_NUM):
    # Get frames
    video = cv2.VideoCapture(FILE_LOCATION+str(i)+FILE_TYPE)

    # Check the size is right.
    if (height and width):
        assert height == int(video.get(4))
        assert width == int(video.get(3))

    # Initialize the parameters.
    frame_num = 0
    loader = []
    height = int(video.get(4))
    width = int(video.get(3))
    #print(height, width, 3)

    # Write the info (FILE_NUM, width, height, color channels) to batch_info.
    if (i == 0):
        if os.path.isdir("dataset") == False:
            os.mkdir("dataset")
        file = open('dataset/batch'+str(BATCH_NUM)+'_info.txt', 'w')
        file.write('{:d} {:d} {:d} {:d}'.format(FILE_NUM, height, width, 3))
        file.close()
    
    # Read video and store it in loader.
    while(True):
        ret, frame = video.read()
        if(ret == False):
            break
        frame_num+=1
        loader.append(frame)
    
    video.release()
    loaders.append(loader)
    loader = []
    
# Write them to the file.
# loader would be a 5-d tensor:
# [number of video in this training set, number of frames, frame height, frame width, color channels]
write_array(loaders, 'dataset/batch'+str(BATCH_NUM))
test_coorectly_stored('dataset/batch'+str(BATCH_NUM)+'_info.txt', 'dataset/batch'+str(BATCH_NUM)+'.npy')