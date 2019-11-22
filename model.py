# -*- coding: utf-8 -*-

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

############################
#         Parameters       #
############################
W_1 = 100
H_1 = 50
W_2 = 120
H_2 = 60
W_3 = 140
H_3 = 70

############################
#           Layers         #
############################
def layer(x, reuse):
    conv1_1 = tf.layers.conv2d(x, 1, 5, padding='same', reuse=reuse)
    pool1_1 = tf.layers.max_pooling2d(input=conv1_1, pool_size=3, strides=2, padding='valid')
    conv1_2 = tf.layers.conv2d(pool1_1, 1, 3, padding='same', reuse=reuse)
    conv1_3 = tf.layers.conv2d(conv1_2, 1, 3, padding='same', reuse=reuse)
    pool1_3 = tf.layers.max_pooling2d(input=conv1_3, pool_size=3, strides=2, padding='valid')
    fc = tf.contrib.layers.flatten(pool1_3)
    logits = tf.layers.dense(fc, 2)
    return logits

def predict(logit, y):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logit))
    pred = tf.argmax(logits, 1)
    correct_pred = tf.equal(tf.cast(pred, tf.int32), y)
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return loss, pred, acc

class Model:
    def __init__(self,
                 learning_rate=0.4,
                 learning_rate_decay_factor=0.99):
        self.x_1 = tf.placeholder(tf.float32, [None, W_1, H_1, 3])
        self.x_2 = tf.placeholder(tf.float32, [None, W_2, H_2, 3])
        self.x_3 = tf.placeholder(tf.float32, [None, W_3, H_3, 3])
        self.y = tf.placeholder(tf.int32, [None])

        self.loss, self.pred, self.acc = self.forward(is_train=True)

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        #self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
        #                            max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def forward(self, is_train, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("model", reuse=reuse):
            # images
            x1 = tf.reshape(self.x_1, shape=[-1, W_1, H_1, 3])
            x2 = tf.reshape(self.x_2, shape=[-1, W_2, H_2, 3])
            x3 = tf.reshape(self.x_3, shape=[-1, W_3, H_3, 3])
            # build layers
            logit_1 = layer(x1, reuse)
            logit_2 = layer(x2, reuse)
            logit_3 = layer(x3, reuse)
        
        # loss...
        loss = [0,0,0]
        pred = [0,0,0]
        acc = [0,0,0]

        loss[0], pred[0], acc[0] = predict(logit_1, self.y)
        loss[1], pred[1], acc[1] = predict(logit_2, self.y)
        loss[2], pred[2], acc[2] = predict(logit_3, self.y)
        
        #Tensorboard
        '''
        if(is_train == True):
            tf.summary.scalar("train_loss", loss)
            tf.summary.scalar("train_accuracy", acc)
        else:
            tf.summary.scalar("val_loss", loss)
            tf.summary.scalar("val_accuracy", acc)
        '''
        return loss, pred, acc # shape: [3], [3], [3]