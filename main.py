import tensorflow as tf
from tensorflow.python.framework import constant_op
from collections import defaultdict

import numpy as np
import sys
import os
import random
random.seed(1229)

from load_data import load_data
from preprocessing import Video

############################
#           Flags          #
############################
tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_integer("symbols", 18430, "vocabulary size.")
tf.app.flags.DEFINE_integer("labels", 5, "Number of labels.")
tf.app.flags.DEFINE_float("learning_rate", 0.005, "learning rate.")
tf.app.flags.DEFINE_integer("epoch", 50, "Number of epoch.")
tf.app.flags.DEFINE_integer("units", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "./dataset", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_integer("per_checkpoint", 1000, "How many steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("inference_version", 0, "The version for inferencing.")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")
FLAGS = tf.app.flags.FLAGS


############################
#         Functions        #
############################
def train(model, sess, dataset):
    st, ed, loss, accuracy = 0, 0, .0, .0
    while ed < len(dataset):
        st, ed = ed, ed+FLAGS.batch_size if ed+FLAGS.batch_size < len(dataset) else len(dataset)
        batch_data = gen_batch_data(dataset[st:ed])
        outputs, summary = model.train_step(sess, batch_data, merged_summary_op) # Tensorboard
        loss += outputs[0]
        accuracy += outputs[1]

    return loss / len(dataset), accuracy / len(dataset)


############################
#           Main           #
############################

# -------- config -------- #
config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True

# ------- run sess ------- #
with tf.Session(config=config) as sess:

    if FLAGS.is_train:
        #print(FLAGS.__flags)
        trX, trY, teX, teY = load_data(FLAGS.data_dir)
        region_cet = [] # shape : [# of video in batch, frames, # of points, 2]
        for i in range(len(trX)):
            v = Video(trX[i], trY[i])
            region_points = v.get_candidate_regions() # shape : [frames, num_points, 2] -> n frame
            region_cet.append(region_points)
        
        cnn_model = Model()
        
        if FLAGS.log_parameters:
            model.print_parameters()
        
        if tf.train.get_checkpoint_state(FLAGS.train_dir):
            print("Reading model parameters from %s" % FLAGS.train_dir)
            model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))

        else:
            print("Created model with fresh parameters.")
            tf.global_variables_initializer().run()
            op_in = model.symbol2index.insert(constant_op.constant(vocab),
                constant_op.constant(list(range(FLAGS.symbols)), dtype=tf.int64))
            sess.run(op_in)
              
        for epoch in list(range(FLAGS.epoch)):
            # TODO: shuffle: random.shuffle(data_train)
            loss_t, accuracy_t = train(model, sess, data_train)
            documant_losses(train_dic, loss_t, accuracy_t, epoch)
            print("epoch %d learning rate %.4f epoch-time %.4f loss %.8f accuracy [%.8f]" % (epoch, model.learning_rate.eval(), time.time()-start_time, loss_t, accuracy_t))
            
            loss_v, accuracy_v = evaluate(model, sess, data_dev, epoch, merged_summary_op) # Tensorboard
            documant_losses(val_dic, loss_v, accuracy_v, epoch)
            print("        dev_set, loss %.8f, accuracy [%.8f]" % (loss_v, accuracy_v))

    else:
        data_dev = load_data(FLAGS.data_dir, 'dev.txt')
        data_test = load_data(FLAGS.data_dir, 'test.txt')

        model = RNN(
                FLAGS.symbols, 
                FLAGS.embed_units,
                FLAGS.units, 
                FLAGS.layers,
                FLAGS.labels,
                embed=None)

        if FLAGS.inference_version == 0:
            model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        else:
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)
        print('restore from %s' % model_path)
        model.saver.restore(sess, model_path)

        # RE
        loss, accuracy = evaluate(model, sess, data_dev, -1, None) # PLOT
        print("        dev_set, loss %.8f, accuracy [%.8f]" % (loss, accuracy))

        inference(model, sess, data_test)

        print("        test_set, write inference results to result.txt")
        
        # write info...
        if os.path.isdir("loss_info") == False:
            os.mkdir("loss_info")
        file = open('loss_info/val_info.txt', 'a')
        file.write('Val Accuracy = {:.8f}\n    Val Loss = {:.8f}\n'.format(accuracy, loss))
