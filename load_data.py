import string
import numpy as np

def get_shape(file_name):
    # -------- loading ------- #
    file = open(file_name+'_info.txt', 'r')
    contents = file.readline()
    file.close()
    shape = []
    for size in contents.split():
        size = size.strip(string.whitespace)
        shape.append(int(size))
    return shape[0], shape[1], shape[2]

def load_data(path):
    print('Creating dataset...')

    trX = None
    trY = None
    teX = None
    teY = None

    for batch_id in range(1): # batch size
        file_name_tr = path+'/batch'+str(batch_id)
        shape = get_shape(file_name_tr)
        trX = np.load(file_name_tr+'.npy', allow_pickle=True)
        trY = np.load(file_name_tr+'_region.npy', allow_pickle=True)

    #file_name_te = path+'batch_test'
    #shape = get_shape(file_name_te)
    #teX = np.load(file_name_te)
    #teY = [1]

    return trX, trY, teX, teY