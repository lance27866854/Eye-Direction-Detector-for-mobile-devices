import numpy as np
import tensorflow as tf

############################
#         Parameters       #
############################
H_3 = 24
W_3 = 48
CONV_1 = 64
CONV_2 = 32
CONV_3 = 16

############################
#           Layers         #
############################
def cnn_layer(x, batch_size, length):
    conv_1 = tf.layers.conv2d(x, CONV_1, 5, padding='same')
    pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=3, strides=2, padding='valid') # shape: batch*length, height, width, depth
    conv_2 = tf.layers.conv2d(pool_1, CONV_2, 3, padding='same')
    pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=2, strides=1, padding='valid') # shape: batch*length, height, width, depth
    conv_3 = tf.layers.conv2d(pool_2, CONV_3, 3, padding='same')
    pool_3 = tf.layers.max_pooling2d(conv_3, pool_size=2, strides=1, padding='valid') # shape: batch*length, height, width, depth
    flat = tf.contrib.layers.flatten(pool_3)
    dense = tf.layers.dense(flat, DENSE) # shape: length, height, width, depth
    
    return dense

def layer_combine(left, right):
    '''
    architecture: CNN + RNN
    '''
    # CNN Left
    dense_l_reshape = cnn_layer(left, batch_size, length)
    # CNN right
    dense_r_reshape = cnn_layer(right, batch_size, length)
    # concat
    concat = tf.concat([dense_l_reshape, dense_r_reshape], axis=2)
    # Fully-connected
    logit = tf.layers.dense(concat, 5)
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
        
        self.loss, self.pred, self.acc = self.forward()
        
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
         # calculate the gradient of parameters
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step, var_list=self.params)
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=3, pad_step_number=True)
        
    def forward(self):
        with tf.variable_scope("model"):
            left = tf.reshape(self.left, shape=[-1, W_3, H_3, 3]) # batch*length, Height, Width, 3
            right = tf.reshape(self.right, shape=[-1, W_3, H_3, 3]) # batch*length, Height, Width, 3
            logit = layer_combine(left, right)

        loss, pred, acc = predict(logit, self.y)
        
        return loss, pred, acc

           