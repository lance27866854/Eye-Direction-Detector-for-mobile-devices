import numpy as np
import tensorflow as tf

############################
#         Parameters       #
############################
H_3 = 24
W_3 = 48
CONV_1 = 64
CONV_2 = 64
CONV_3 = 64
DENSE = 300

############################
#           Layers         #
############################
def layer_combine(left, right, is_train, reuse):
    '''
    architecture: CNN
    '''
    with tf.variable_scope("se_model", reuse=reuse):
        # Left
        conv_l1 = tf.layers.conv2d(left, CONV_1, 3, padding='same', name='se_convl1', reuse=reuse)
        bn_l1 = tf.layers.batch_normalization(conv_l1, training=is_train, name='se_bnl1')
        relu_l1 = tf.nn.relu(bn_l1)
        pool_l1 = tf.layers.max_pooling2d(relu_l1, pool_size=2, strides=2, padding='valid') # shape: batch*length, height, width, depth

        conv_l2 = tf.layers.conv2d(pool_l1, CONV_2, 3, padding='same', name='se_convl2', reuse=reuse)
        bn_l2 = tf.layers.batch_normalization(conv_l2, training=is_train, name='se_bnl2')
        relu_l2 = tf.nn.relu(bn_l2)
        pool_l2 = tf.layers.max_pooling2d(relu_l2, pool_size=2, strides=2, padding='valid') # shape: batch*length, height, width, depth
        
        conv_l3 = tf.layers.conv2d(pool_l2, CONV_3, 3, padding='same', name='se_convl3', reuse=reuse)
        bn_l3 = tf.layers.batch_normalization(conv_l3, training=is_train, name='se_bnl3')
        relu_l3 = tf.nn.relu(bn_l3)
        pool_l3 = tf.layers.max_pooling2d(relu_l3, pool_size=2, strides=2, padding='valid') # shape: batch*length, height, width, depth
        
        flat_l = tf.contrib.layers.flatten(pool_l3)
        dense_l_reshape = tf.layers.dense(flat_l, DENSE, name='se_densel', reuse=reuse) # shape: length, height, width, depth

        # Right
        conv_r1 = tf.layers.conv2d(right, CONV_1, 3, padding='same', name='se_convr1', reuse=reuse)
        bn_r1 = tf.layers.batch_normalization(conv_r1, training=is_train, name='se_bnr1')
        relu_r1 = tf.nn.relu(bn_r1)
        pool_r1 = tf.layers.max_pooling2d(relu_r1, pool_size=2, strides=2, padding='valid') # shape: batch*length, height, width, depth

        conv_r2 = tf.layers.conv2d(pool_r1, CONV_2, 3, padding='same', name='se_convr2', reuse=reuse)
        bn_r2 = tf.layers.batch_normalization(conv_r2, training=is_train, name='se_bnr2')
        relu_r2 = tf.nn.relu(bn_r2)
        pool_r2 = tf.layers.max_pooling2d(relu_r2, pool_size=2, strides=2, padding='valid') # shape: batch*length, height, width, depth
        
        conv_r3 = tf.layers.conv2d(pool_r2, CONV_3, 3, padding='same', name='se_convr3', reuse=reuse)
        bn_r3 = tf.layers.batch_normalization(conv_r3, training=is_train, name='se_bnr3')
        relu_r3 = tf.nn.relu(bn_r3)
        pool_r3 = tf.layers.max_pooling2d(relu_r3, pool_size=2, strides=2, padding='valid') # shape: batch*length, height, width, depth
        
        flat_r = tf.contrib.layers.flatten(pool_r3)
        dense_r_reshape = tf.layers.dense(flat_r, DENSE, name='se_denser', reuse=reuse) # shape: length, height, width, depth

        # Fully-connected
        concat = tf.concat([dense_l_reshape, dense_r_reshape], axis=1)
        logit = tf.layers.dense(concat, 5, name='se_dense_combine', reuse=reuse)

    return logit

def predict(logit, y):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logit))
    pred = tf.argmax(logit, 1)
    correct_pred = tf.equal(tf.cast(pred, tf.int32), y)
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return loss, pred, acc

############################
#           Model          #
############################
class Model(object):
    def __init__(self,
                learning_rate=0.005,
                max_gradient_norm=5.0):

        self.left = tf.placeholder(tf.float32, [None, H_3, W_3, 3]) # batch, length, Height, Width, 3
        self.right = tf.placeholder(tf.float32, [None, H_3, W_3, 3]) # batch, length, Height, Width, 3
        self.y = tf.placeholder(tf.int32, [None]) # batch
        
        self.loss, self.pred, self.acc = self.forward(is_train=True, reuse=None)
        self.loss_val, self.pred_val, self.acc_val = self.forward(is_train=False, reuse=True)
        
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # calculate the gradient of parameters
        with tf.control_dependencies(self.update_op):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step, var_list=self.params)
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=3, pad_step_number=True)
        
    def forward(self, is_train, reuse=None):
        left = tf.reshape(self.left, shape=[-1, W_3, H_3, 3]) # batch*length, Height, Width, 3
        right = tf.reshape(self.right, shape=[-1, W_3, H_3, 3]) # batch*length, Height, Width, 3
        logit = layer_combine(left, right, is_train=is_train, reuse=reuse)

        loss, pred, acc = predict(logit, self.y)
        
        return loss, pred, acc

           