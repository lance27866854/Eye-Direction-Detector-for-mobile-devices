import tensorflow as tf
############################
#         Parameters       #
############################
W_1 = 24
H_1 = 12
W_2 = 36
H_2 = 18
W_3 = 48
H_3 = 24
CONV_1 = 64
CONV_2 = 64
CONV_3 = 64

############################
#           Layers         #
############################
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
                 learning_rate_decay_factor=0.95):
        self.x_1 = tf.placeholder(tf.float32, [None, H_1, W_1, 3])
        self.x_2 = tf.placeholder(tf.float32, [None, H_2, W_2, 3])
        self.x_3 = tf.placeholder(tf.float32, [None, H_3, W_3, 3])
        self.y = tf.placeholder(tf.int32, [None])

        self.loss_1, self.pred_1, self.acc_1, self.loss_2, self.pred_2, self.acc_2, self.loss_3, self.pred_3, self.acc_3, self.logit_1, self.logit_2, self.logit_3 = self.forward(is_train=True, reuse=None)
        self.val_loss_1, self.val_pred_1, self.val_acc_1, self.val_loss_2, self.val_pred_2, self.val_acc_2, self.val_loss_3, self.val_pred_3, self.val_acc_3, self.val_logit_1, self.val_logit_2, self.val_logit_3 = self.forward(is_train=False, reuse=True)

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        
        self.update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_op):
            self.train_op1 = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_1, global_step=self.global_step, var_list=self.params)
            self.train_op2 = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_2, global_step=self.global_step, var_list=self.params)
            self.train_op3 = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_3, global_step=self.global_step, var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=1, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def forward(self, is_train, reuse):
        with tf.variable_scope("model", reuse=reuse):
            # images
            x1 = tf.reshape(self.x_1, shape=[-1, W_1, H_1, 3])
            x2 = tf.reshape(self.x_2, shape=[-1, W_2, H_2, 3])
            x3 = tf.reshape(self.x_3, shape=[-1, W_3, H_3, 3])

            # build layers
            # Layer 1
            conv1_1 = tf.layers.conv2d(x1, CONV_1, 5, padding='same', name='conv1_1', reuse=reuse)
            bn1_1 = tf.layers.batch_normalization(conv1_1, training=is_train)
            relu1_1 = tf.nn.relu(bn1_1)
            pool1_1 = tf.layers.max_pooling2d(relu1_1, pool_size=3, strides=2, padding='valid')
            
            conv1_2 = tf.layers.conv2d(pool1_1, CONV_2, 3, padding='same', name='conv1_2', reuse=reuse)
            bn1_2 = tf.layers.batch_normalization(conv1_2, training=is_train)
            relu1_2 = tf.nn.relu(bn1_2)
            pool1_2 = tf.layers.max_pooling2d(relu1_2, pool_size=3, strides=2, padding='valid')
            
            conv1_3 = tf.layers.conv2d(pool1_2, CONV_3, 3, padding='same', name='conv1_3', reuse=reuse)
            bn1_3 = tf.layers.batch_normalization(conv1_3, training=is_train)
            relu1_3 = tf.nn.relu(bn1_3)
            pool1_3 = tf.layers.max_pooling2d(relu1_3, pool_size=2, strides=2, padding='valid')

            fc1 = tf.contrib.layers.flatten(pool1_3)
            logit_1 = tf.layers.dense(fc1, 3, name='dense1_1', reuse=reuse)

            # Layer 2
            conv2_1 = tf.layers.conv2d(x2, CONV_1, 5, padding='same', name='conv2_1', reuse=reuse)
            bn2_1 = tf.layers.batch_normalization(conv2_1, training=is_train)
            relu2_1 = tf.nn.relu(bn2_1)
            pool2_1 = tf.layers.max_pooling2d(relu2_1, pool_size=3, strides=2, padding='valid')
            
            conv2_2 = tf.layers.conv2d(pool2_1, CONV_2, 3, padding='same', name='conv2_2', reuse=reuse)
            bn2_2 = tf.layers.batch_normalization(conv2_2, training=is_train)
            relu2_2 = tf.nn.relu(bn2_2)
            pool2_2 = tf.layers.max_pooling2d(relu2_2, pool_size=2, strides=1, padding='valid')
            
            conv2_3 = tf.layers.conv2d(pool2_2, CONV_3, 3, padding='same', name='conv2_3', reuse=reuse)
            bn2_3 = tf.layers.batch_normalization(conv2_3, training=is_train)
            relu2_3 = tf.nn.relu(bn2_3)
            pool2_3 = tf.layers.max_pooling2d(relu2_3, pool_size=2, strides=1, padding='valid')

            fc2 = tf.contrib.layers.flatten(pool2_3)
            logit_2 = tf.layers.dense(fc1, 3, name='dense2_1', reuse=reuse)
            
            # Layer 3
            conv3_1 = tf.layers.conv2d(x3, CONV_1, 5, padding='same', name='conv3_1', reuse=reuse)
            bn3_1 = tf.layers.batch_normalization(conv3_1, training=is_train)
            relu3_1 = tf.nn.relu(bn3_1)
            pool3_1 = tf.layers.max_pooling2d(relu3_1, pool_size=3, strides=2, padding='valid')
            
            conv3_2 = tf.layers.conv2d(pool3_1, CONV_2, 3, padding='same', name='conv3_2', reuse=reuse)
            bn3_2 = tf.layers.batch_normalization(conv3_2, training=is_train)
            relu3_2 = tf.nn.relu(bn3_2)
            pool3_2 = tf.layers.max_pooling2d(relu3_2, pool_size=2, strides=1, padding='valid')
            
            conv3_3 = tf.layers.conv2d(pool3_2, CONV_3, 3, padding='same', name='conv3_3', reuse=reuse)
            bn3_3 = tf.layers.batch_normalization(conv3_3, training=is_train)
            relu3_3 = tf.nn.relu(bn3_3)
            pool3_3 = tf.layers.max_pooling2d(relu3_3, pool_size=2, strides=1, padding='valid')

            fc3 = tf.contrib.layers.flatten(pool3_3)
            logit_3 = tf.layers.dense(fc3, 3, name='dense3_1', reuse=reuse)

        # loss...
        loss_1, pred_1, acc_1 = predict(logit_1, self.y)
        loss_2, pred_2, acc_2 = predict(logit_2, self.y)
        loss_3, pred_3, acc_3 = predict(logit_3, self.y)

        return loss_1, pred_1, acc_1, loss_2, pred_2, acc_2, loss_3, pred_3, acc_3, tf.nn.softmax(logit_1), tf.nn.softmax(logit_2), tf.nn.softmax(logit_3) 
        