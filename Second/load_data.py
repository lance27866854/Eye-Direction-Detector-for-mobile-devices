import string
import numpy as np

def get_shape(path):
    print(path)
    # -------- loading ------- #
    file = open(path+'_info.txt', 'r')
    contents = file.readline()
    
    shape = []
    for size in contents.split():
        size = size.strip(string.whitespace)
        shape.append(int(size))
    gt = []
    for i in range(1):#shape[0]
        contents = file.readline()
        gt.append([int(contents[0])])
    file.close()

    return shape, np.array(gt)

def load_data(path, batch_id=0):
    print('Creating dataset...')

    file_name1 = path+'_RNN/batch'+str(batch_id)
    file_name2 = path+'/batch'+str(batch_id)
    shape, trY = get_shape(file_name2)
    trX = np.load(file_name1+'.npy', allow_pickle=True)
    print(trX.shape, trY.shape)
    return trX, trY, shape