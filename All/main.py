from model_first import Model as model_f
from model_second import Model as model_s
from utils import get_first_data, get_eye, get_label

with tf.Session(config=config) as sess:
    
    first_model = model_f()
    second_model = model_s()

    first_model.saver.restore(sess, 'first_model')
    second_model.saver.restore(sess, 'second_model')

    video = np.load('video.npy', allow_pickle=True)
    for frame in  video:
        data_1, data_2, data_3, candidate_list = get_first_data(frame)
        left_eye, right_eye = get_eye(model, sess, data_1, data_2, data_3, candidate_list)
        label = get_label(model, sess, left_eye, right_eye)
        print(label)
    