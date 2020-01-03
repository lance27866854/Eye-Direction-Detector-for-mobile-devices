#-*-coding:utf-8-*-
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('C:\\Users\\user\\Desktop\\Model\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\user\\Desktop\\Model\\haarcascade_eye.xml')
 
# get images
videos = np.load('dataset/0/video.npy', allow_pickle=True)
region_points = np.load('dataset/0/region_point.npy')

def detect_eye(img, region_point, num, threshold_left=150, threshold_right=150):
    # if detected
    have_left = False
    have_right = False

    # convert to grey
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, w, h) in faces:
        #img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = img[y: y + h, x: x + w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        #if len(eyes) <= 2:
        for(ex, ey, ew, eh) in eyes:
            X = x+ex+ew/2
            Y = y+ey+eh/2
            left_dis = pow(X-region_point[0][0], 2)+pow(Y-region_point[0][1], 2)
            right_dis = pow(X-region_point[1][0], 2)+pow(Y-region_point[1][1], 2)
            
            if left_dis < threshold_left:
                have_left = True

            if left_dis < threshold_left:
                have_right = True

            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),( 0 , 255 , 0 ), 2 )
        
        b,g,r = cv2.split(img)  
        img2 = cv2.merge([r,g,b])
        cv2.imshow("img", img2)
        cv2.waitKey( 0 )

    return have_left, have_right


def detect_video(videos, region_points):
    all_left = 0
    all_right = 0
    correct_left = 0
    correct_right = 0

    for i in range(len(videos)):
        print("processing "+str(i)+"-th video...")
        num = 0
        rp = region_points[i]
        if rp[0][0]!=-1:
            num += 1
        if rp[0][1]!=-1:
            num += 1

        for j in range(len(videos[i])):
            img = videos[i][j]    
            have_left, have_right = detect_eye(img, rp, num)

            if rp[0][0]!=-1:
                all_left = all_left+1
                correct_left = correct_left+1 if have_left else correct_left

            if rp[1][0]!=-1:
                all_right = all_right+1
                correct_right = correct_right+1 if have_right else correct_right
            break
    
    return all_left, all_right, correct_left, correct_right


videos = np.load('dataset/test_unknown/video.npy', allow_pickle=True)
region_points = np.load('dataset/test_unknown/region_point.npy')
all_left, all_right, correct_left, correct_right = detect_video(videos, region_points)
print(all_left, all_right, correct_left, correct_right)

