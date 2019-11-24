import string
import numpy as np
from preprocessing import Video

def get_shape(path):
    print(path)
    # -------- loading ------- #
    file = open(path+'_info.txt', 'r')
    contents = file.readline()
    file.close()
    shape = []
    for size in contents.split():
        size = size.strip(string.whitespace)
        shape.append(int(size))
    return shape[0], shape[1], shape[2]

def load_data(path, batch_id=0):
    print('Creating dataset...')
    
    file_name_tr = path+'/batch'+str(batch_id)
    shape = get_shape(file_name_tr)
    trX = np.load(file_name_tr+'.npy', allow_pickle=True)
    trY = np.load(file_name_tr+'_region.npy', allow_pickle=True)

    return trX, trY, teX, teY, shape

def point_convertor(dir):
    trX, trY, teX, teY, shape = load_data(dir)
    region_cet = [] # shape : [num_points, 4]
    for i in range(1): # len(trX) -> all
        print("Processing the "+str(i)+"-th video...")
        v = Video(trX[1], trY[1], 1)
        v.get_candidate_regions(region_cet)

    data = np.array(region_cet)
    np.save("../tool/dataset/data_points.npy", data)
    return data, trX, shape

def get_raw_points(dir):
    trX, trY, teX, teY, shape = load_data(dir)
    region_cet = []# shape : [video, frame, n, 2]
    for i in range(1): # len(trX) -> all
        print("Processing the "+str(i)+"-th video...")
        v = Video(trX[1], trY[1], 1)
        frame_region = v.get_raw_candidate_regions()
        region_cet.append(frame_region)

    return region_cet, trX, shape