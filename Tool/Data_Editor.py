import cv2
import numpy as np
import string
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

############################
#        Data Editor       #
############################
class Data_Editor:
    def __init__(self, name, answer, path, file_num=2, width=180, height=360, eye_W=64, eye_H=32):
        self.name = name
        self.file_num = file_num
        self.width = width
        self.height = height
        self.eye_W = eye_W
        self.eye_H = eye_H
        self.path = path

        # Write the info (FILE_NUM, width, height, color channels) to batch_info.
        if os.path.isdir(self.name) == False:
            os.mkdir(self.name)
        file = open(self.name+'/info.txt', 'w')
        file.write('{:d} {:d} {:d} {:d}\n'.format(self.file_num, self.height, self.width, 3))
        file.close()

        np.save(self.name+'/ground_truth', answer) 
        
    # -------- Video Vector -------- #
    # ENTRY
    def convert_videos_to_files(self):
        # Convert the files (videos) to arrays.
        loaders = []

        for i in range(self.file_num):
            # Get frames
            # CHANGE HERE
            video = cv2.VideoCapture(self.path+str(i)+'.mp4')

            # Initialize the parameters.
            frame_num = 0
            loader = []
            
            # Read video and store it in loader.
            while(True):
                ret, frame = video.read()
                if(ret == False):
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_HSV = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                frame_num+=1
                if frame.shape[0]>2*frame.shape[1]:
                    st = (int)((frame.shape[0]-2*frame.shape[1])/2)
                    ed = st+2*frame.shape[1]
                    frame = frame[st:ed, :]
                elif frame.shape[1]*2>frame.shape[0]:
                    st = (int)((frame.shape[1]*2-frame.shape[0])/2)
                    ed = (int)(st+frame.shape[0]/2)
                    frame = frame[:, st:ed]
                loader.append(cv2.resize(frame, dsize=(self.width, self.height), interpolation=cv2.INTER_AREA))
            
            video.release()
            loaders.append(loader)
            loader = []
            
        # Write them to the file.
        # loader would be a 5-d tensor:
        # [number of video in this training set, number of frames, frame height, frame width, color channels]
        np.save(self.name+'/video', loaders)
        self.test_coorectly_stored()
    
    def test_coorectly_stored(self):
        file = open(self.name+'/info.txt', 'r')
        contents = file.readline()
        file.close()
        
        shape_assert = []
        for word in contents.split():
            word = word.strip(string.whitespace)
            shape_assert.append(int(word))

        data = np.load(self.name+'/video.npy', allow_pickle=True)
        assert len(data) == shape_assert[0]
        assert data[0][0].shape[0] == shape_assert[1]
        assert data[0][0].shape[1] == shape_assert[2]
        assert data[0][0].shape[2] == shape_assert[3]

    # -------- Position Vector ------- #
    # ENTRY
    def convert_eye_points_to_files(self, point, store=False):
        np.save(self.name+'/region_point', point)

    def show_plot(self, frame, point):
        fig,ax = plt.subplots(1)
        ax.imshow(frame)
        rect_l = patches.Rectangle((point[0][0]-self.eye_W/2, point[0][1]-self.eye_H/2), self.eye_W, self.eye_H, linewidth=1, edgecolor='r', facecolor='none')
        rect_r = patches.Rectangle((point[1][0]-self.eye_W/2, point[1][1]-self.eye_H/2), self.eye_W, self.eye_H, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect_l)
        ax.add_patch(rect_r)
        plt.show()

    def random_show(self, point, i):# point=[[1,1], [10,10]]
        data = np.load(self.name+'/video.npy', allow_pickle=True)
        gt = np.load(self.name+'/ground_truth.npy') 
        video = data[i]
        fi = random.randrange(len(data[i]))
        frame = video[fi]
        self.show_plot(frame, point)
        print(gt[i])

    def test(self):
        videos = np.load(self.name+'/video.npy', allow_pickle=True)
        region_points = np.load(self.name+'/region_point.npy')
        for i in range(len(videos)):
            self.random_show(region_points[i], i)