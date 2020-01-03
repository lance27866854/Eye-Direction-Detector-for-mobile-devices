import numpy as np

videos1 = np.load('4/video.npy', allow_pickle=True)
region_points1 = np.load('4/region_point.npy')
gt1 = np.load('4/ground_truth.npy')

videos2 = np.load('5/video.npy', allow_pickle=True)
region_points2 = np.load('5/region_point.npy')
gt2 = np.load('5/ground_truth.npy')

n_v = []
n_r = []
n_g = []

for i in range(len(videos1)):
    n_v.append(videos1[i])
for i in range(len(videos2)):
    n_v.append(videos2[i])

for i in range(len(region_points1)):
    n_r.append(region_points1[i])
for i in range(len(region_points2)):
    n_r.append(region_points2[i])

for i in range(len(gt1)):
    n_g.append(gt1[i])
for i in range(len(gt2)):
    n_g.append(gt2[i])

print(len(n_v))

print(len(n_r))

print(len(n_g))

np.save('new/video', n_v)
np.save('new/region_point', n_r)
np.save('new/ground_truth', n_g)