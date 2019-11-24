# -*- coding: utf-8 -*-

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

############################
#         Parameters       #
############################
W_1 = 140
H_1 = 70
W_2 = 160
H_2 = 80
W_3 = 180
H_3 = 90

############################
#           Layers         #
############################
def layer(x):
    conv1_1 = tf.layers.conv2d(x, 1, 5, padding='same')
    pool1_1 = tf.layers.max_pooling2d(conv1_1, pool_size=3, strides=2, padding='valid')
    conv1_2 = tf.layers.conv2d(pool1_1, 1, 3, padding='same')
    conv1_3 = tf.layers.conv2d(conv1_2, 1, 3, padding='same')
    pool1_3 = tf.layers.max_pooling2d(conv1_3, pool_size=3, strides=2, padding='valid')
    fc = tf.contrib.layers.flatten(pool1_3)
    logits = tf.layers.dense(fc, 3)
    return logits

def predict(logit, y):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logit))
    pred = tf.argmax(logit, 1)
    correct_pred = tf.equal(tf.cast(pred, tf.int32), y)
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return loss, pred, acc

############################
#           Model          #
############################
class Model:
    def __init__(self,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.99):
        self.x_1 = tf.placeholder(tf.float32, [None, H_1, W_1, 3])
        self.x_2 = tf.placeholder(tf.float32, [None, H_2, W_2, 3])
        self.x_3 = tf.placeholder(tf.float32, [None, H_3, W_3, 3])
        self.y = tf.placeholder(tf.int32, [None])

        self.loss_1, self.pred_1, self.acc_1, self.loss_2, self.pred_2, self.acc_2, self.loss_3, self.pred_3, self.acc_3, self.logit_1, self.logit_2, self.logit_3= self.forward()

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op1 = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_1, global_step=self.global_step, var_list=self.params)
        self.train_op2 = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_2, global_step=self.global_step, var_list=self.params)
        self.train_op3 = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_3, global_step=self.global_step, var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def forward(self):
        with tf.variable_scope("model"):
            # images
            x1 = tf.reshape(self.x_1, shape=[-1, W_1, H_1, 3])
            x2 = tf.reshape(self.x_2, shape=[-1, W_2, H_2, 3])
            x3 = tf.reshape(self.x_3, shape=[-1, W_3, H_3, 3])
            # build layers
            logit_1 = layer(x1)
            logit_2 = layer(x2)
            logit_3 = layer(x3)
        
        # loss...
        loss_1, pred_1, acc_1 = predict(logit_1, self.y)
        loss_2, pred_2, acc_2 = predict(logit_2, self.y)
        loss_3, pred_3, acc_3 = predict(logit_3, self.y)

        return loss_1, pred_1, acc_1, loss_2, pred_2, acc_2, loss_3, pred_3, acc_3, tf.nn.softmax(logit_1), tf.nn.softmax(logit_2), tf.nn.softmax(logit_3) 
        