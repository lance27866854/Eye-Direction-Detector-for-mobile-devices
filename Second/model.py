import numpy as np
import tensorflow as tf

############################
#         Parameters       #
############################
W_1 = 140
H_1 = 70
DENSE = 128
R_UNIT = 64

############################
#           Layers         #
############################
def layer(x):
    # CNN
    conv = tf.layers.conv2d(x, 1, 5, padding='same')
    pool = tf.layers.max_pooling2d(conv, pool_size=3, strides=2, padding='valid') # shape: batch, height, width, depth
    flat = tf.contrib.layers.flatten(pool)
    dense = tf.layers.dense(flat, DENSE) # shape: length, height, width, depth
    dense_ = tf.reshape(dense, shape=[1, tf.shape(dense)[0], DENSE])
    # RNN
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=R_UNIT)
    outputs, last_states = tf.nn.dynamic_rnn(cell=cell, inputs=dense_, dtype=tf.float32, sequence_length=None)
    # Fully-connected
    logit = tf.layers.dense(outputs, 5)

    return logit

def predict(logit, y):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logit[0]))
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

        self.x = tf.placeholder(tf.float32, [None, H_1, W_1, 3]) # batch, length, Height, Width, 3
        self.y = tf.placeholder(tf.int32, [1])
        
        self.loss, self.pred, self.acc = self.forward()
        
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
         # calculate the gradient of parameters
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients = tf.gradients(self.loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=3, pad_step_number=True)
        
    def forward(self):
        with tf.variable_scope("model"):
            x = tf.reshape(self.x, shape=[-1, W_1, H_1, 3]) # length, Height, Width, 3
            # build layers
            logit = layer(x)

        loss, pred, acc = predict(logit, self.y)
        
        return loss, pred, acc

           