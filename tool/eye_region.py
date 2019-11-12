import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import cv2

# PARAMETERS
BATCH_NUM = 0
FILE_LOCATION = 'dataset\\'
FILE_TYPE = '.npy'
WRITE = False

# tool functions
def show_plot(frame, left_rect, right_rect):
    fig,ax = plt.subplots(1)
    ax.imshow(frame)
    rect_l = patches.Rectangle((left_rect[0], left_rect[1]), left_rect[2], left_rect[3], linewidth=1, edgecolor='r', facecolor='none')
    rect_r = patches.Rectangle((right_rect[0], right_rect[1]), right_rect[2], right_rect[3], linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect_l)
    ax.add_patch(rect_r)
    plt.show()

def random_show_all(loaders, data):
    for i in range(len(data)):
        fi = random.randrange(len(data[i]))
        frame = video[fi]
        left_rect = loaders[i][0] # [x, y, width, height]
        right_rect =  loaders[i][1] # [x, y, width, height]
        show_plot(frame, left_rect, right_rect)

def random_show(loaders, data, i):
    video = data[i]
    fi = random.randrange(len(data[i]))
    frame = video[fi]
    left_rect = loaders[i][0] # [x, y, width, height]
    right_rect =  loaders[i][1] # [x, y, width, height]
    show_plot(frame, left_rect, right_rect)

# Loader
# loader.shape = [number of videos, 2, 4]
loaders = [[[153, 367, 80, 30], [303, 357, 80, 30]]]

# process data
if WRITE == True:
    np.save(FILE_LOCATION+'batch'+str(BATCH_NUM)+'_region'+FILE_TYPE, loaders)
    
else: 
    data = np.load(FILE_LOCATION+'batch'+str(BATCH_NUM)+FILE_TYPE)
    # take one random frame from the i-th video.
    random_show(loaders, data, 0)
    #random_show_all(loaders, data)