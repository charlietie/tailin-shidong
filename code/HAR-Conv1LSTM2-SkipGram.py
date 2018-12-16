""" Build Conv1 + LSTM2 architecture
    Tailin Chen
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time

import numpy as np
import tensorflow as tf
import tensorflow.layers as layers

from tensorflow.nn import rnn_cell

from tensorflow.keras.datasets.cifar10 import *
from tensorflow.keras.utils import to_categorical

from utils import load_dataset, op_sliding_window, get_batch, make_dir

"""
Step 1. Read in data and preprocess the data
"""

# Data parameters
N_CLASSES = 18
WINDOW_LENGTH = 64
SLIDING_STRIDE = 4
N_DIMENSION = 113
N_CHANNEL = 1

# Model parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIP_STEP = 10
DROPOUT = 0.75
N_EPOCHS = 10

n_hidden_lstm_1 = 128
n_hidden_lstm_2 = 256

# using pre-built (in utils.py) function to load in data

# PC
# X_train, y_train, X_test, y_test = load_dataset("opportunity",
#                                                 dataset_file_path="C:/Users/41762/Dropbox/MSc-Dissertation/Dataset/Opportunity/wearable_113_ambient_40_gestures_new.data")

# MAC
X_train, y_train, X_test, y_test = load_dataset("opportunity",
                                                dataset_file_path="/Users/chentailin/Dropbox/MSc-Dissertation/Dataset/Opportunity/wearable_113_ambient_40_gestures_new.data")

# Select only the 113 wearable sensors for researching.
X_train = X_train[:, :N_DIMENSION]
X_test = X_test[:, :N_DIMENSION]

# Operating sliding window with respect to Time "T = WINDOW_LENGTH"
X_train, y_train = op_sliding_window(X_train, y_train, WINDOW_LENGTH, SLIDING_STRIDE)
X_test, y_test = op_sliding_window(X_test, y_test, WINDOW_LENGTH, SLIDING_STRIDE)

# Reshape the X_train, X_test to shape [None，WINDOW_LENGTH, N_DIMENSION, 1]
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

print("X_train shape is ", X_train.shape)  # X_train shape is  (139475, 64, 113, 1)
print("X_test shape is ", X_test.shape)  # X_test shape is  (29672, 64, 113, 1)

# Operating One-hot encoder on the label
y_train = to_categorical(y_train, N_CLASSES)
y_test = to_categorical(y_test, N_CLASSES)

print("y_train shape is ", y_train.shape)  # y_train shape is  (139475, 18)
print("y_test shape is ", y_test.shape)  # y_test shape is  (29672, 18)

"""
Step 2. Create placeholders for features and labels
"""
# Each sample is in the shape of  (1 , WINDOW_LENGTH , N_DIMENSION) (1, 64, 113), therefore, Conv will be operated on this shape
#
# We'll be doing dropout for hidden layer so we'll need a placeholder
# for the dropout probability too
# Use None for shape so we can change the batch_size once we've built the graph

with tf.name_scope('Input_data'):
    X = tf.placeholder(tf.float32, [None, WINDOW_LENGTH, N_DIMENSION, N_CHANNEL], 'X_placeholder')
    Y = tf.placeholder(tf.float32, [None, N_CLASSES], name="Y_placeholder")

#
dropout = tf.placeholder(tf.float32, name='dropout')

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')  # global step is a flag use variable

"""
The model parameters is defined as:

hybrid = modelParam(
  'Hybrid',
  {
   "nb_conv_blocks" : 1,
   "nb_conv_kernels" : [50],
   "conv_kernels_size" : [(11,1)],
   "pooling_size" : [(2,1)],
   "conv_activation" : 'relu',
   "nb_lstm_layers": 2,
   "lstm_output_dim" : [600,600],
   "dense_size" : [512],
   "dense_activation" : 'relu',
  }
)
"""
with tf.variable_scope('conv1') as scope:
    conv1 = tf.layers.conv2d(inputs=X,
                             filters=50,
                             kernel_size=[11, 1],
                             strides=[1, 1],
                             padding='SAME',
                             activation=tf.nn.relu,
                             name='conv1')

    # output is of dimension BATCH_SIZE x 28 x 28 x 32
    # conv1 = layers.conv2d(X, 32, 5, 3, activation_fn=tf.nn.relu, padding='SAME')

with tf.variable_scope('pool1') as scope:
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2, 1],
                                    strides=[2, 1],
                                    name='pool1',
                                    padding='SAME')
with tf.name_scope("Reshape_last_two_dimensions"):
    """
    tihs is to reshape the inputs with respect to time T
    """
    reshape = tf.reshape(pool1, shape=(BATCH_SIZE, pool1.shape[1], pool1.shape[2] * pool1.shape[3]))

"""
添加skip gram 模型
"""
with tf.name_scope("feature_unstack"):
    unstacked_feature = tf.unstack(reshape, axis=1)
    # unpacked_feature = tf.unpack

with tf.Session() as sess:
    # Run Initializer for all the variables
    sess.run(tf.global_variables_initializer())

    start_time = time.time()

    # start feed the data and train the model
    X_batch, Y_batch = get_batch(X_train, y_train, batch_size=BATCH_SIZE, index=1)

    a = sess.run(unstacked_feature,
                             feed_dict={X: X_batch})
    print(a)
    print("  time: {} seconds".format(time.time() - start_time))

# with tf.variable_scope('lstm_weights_and_biases'):
#     weights = {
#         # (32,5650)
#         'in': tf.get_variable(name='in_weights',
#                               shape=[reshape.shape[2], n_hidden_lstm_1],
#                               initializer=tf.random_normal_initializer()),
#
#         'out': tf.get_variable(name='out_weights',
#                                shape=[n_hidden_lstm_1, N_CLASSES])
#     }
#
#     biases = {
#         # (128,)
#         'in': tf.get_variable(name='in_biases',
#                               shape=[n_hidden_lstm_1, ],
#                               initializer=tf.constant_initializer(0.1)),
#         'out': tf.get_variable(name='out_biases',
#                                shape=[N_CLASSES, ],
#                                initializer=tf.constant_initializer(0.1))
#
#     }
#
#
# def RNN(X, weights, biases):
#     n_inputs = X.shape[2]
#     n_steps = X.shape[1]
#
#     X = tf.reshape(X, [-1, n_inputs])
#
#     X_in = tf.matmul(X, weights['in'] + biases['in'])
#     X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_lstm_1])
#
#     lstm_cell = rnn_cell.BasicLSTMCell(num_units=n_hidden_lstm_1,
#                                        forget_bias=1.0,
#                                        state_is_tuple=True
#                                        )
#     # lstm cell is divided into two parts (c_state主线, m_state分线)
#     _init_state = lstm_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)
#
#     outputs, states = tf.nn.dynamic_rnn(lstm_cell,
#                                         X_in,
#                                         initial_state=_init_state,
#                                         time_major=False
#                                         )
#
#     results = tf.matmul(states[1], weights['out']) + biases['out']
#
#     return results
#
#
# with tf.name_scope("logits"):
#     logits = RNN(reshape, weights, biases)
#
# with tf.name_scope('prediction_while_training'):
#     train_preds_op = tf.argmax(logits, axis=1)
#     actual_pred_op = tf.argmax(Y, axis=1)
#
# with tf.name_scope("accuracy"):
#     correct_batch_predicttion = tf.equal(train_preds_op, actual_pred_op)
#     batch_accuracy = tf.reduce_mean(tf.cast(correct_batch_predicttion, tf.float32))
#
# with tf.name_scope("cross_entropy_loss"):
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
#
# with tf.name_scope("summaries"):
#     # summary the loss, accuracy on the tensorboard
#     tf.summary.scalar("batch_loss", loss)
#     tf.summary.scalar("batch_accuracy", batch_accuracy)
#     summary_op = tf.summary.merge_all()
#
# with tf.name_scope("Adam_Optimizer"):
#     optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)
#
# # make the checkpoints dir
# make_dir("checkpoints")
# make_dir("checkpoints/HAR-Conv1LSTM2")
#
# # train model
#
# with tf.Session() as sess:
#     # Run Initializer for all the variables
#     sess.run(tf.global_variables_initializer())
#
#     # Instantiate a Saver() to save the model states
#     saver = tf.train.Saver()
#
#     # make a writer to visualize the model state
#     writer = tf.summary.FileWriter('graphs/HAR-Conv1LSTM2', sess.graph)
#
#     ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/HAR-Conv1LSTM2/checkpoints'))
#
#     # if that checkpoint exists, restore from  checkpoint
#     if ckpt and ckpt.model_checkpoint_path:
#         saver.restore(sess, ckpt.model_checkpoint_path)
#
#     initial_step = global_step.eval()  # global_step.eval() is 0
#
#     start_time = time.time()
#
#     n_batches = int(X_train.shape[0] / BATCH_SIZE)
#     batch_flag = 0
#
#     total_loss = 0.0
#
#     print("One epoch contains {} steps.".format(n_batches))
#
#     for index in range(initial_step, n_batches * N_EPOCHS):
#
#         X_batch, Y_batch = get_batch(X_train, y_train, batch_size=BATCH_SIZE, index=batch_flag)
#
#         # control the data reader
#         if batch_flag >= n_batches - 1:
#             batch_flag = 0
#         batch_flag += 1
#
#         # start feed the data and train the model
#
#         _, batch_loss, accuracy, summary = sess.run([optimizer, loss, batch_accuracy, summary_op],
#                                                     feed_dict={X: X_batch, Y: Y_batch, dropout: DROPOUT})
#
#         writer.add_summary(summary=summary, global_step=index)
#
#         # sum the batch loss for the total loss
#         total_loss += batch_loss
#
#         if (index + 1) % SKIP_STEP == 0:
#             print("Average loss at step {} : {:5.1f}".format(index + 1,
#                                                              total_loss / SKIP_STEP))
#
#             print("Average accuracy at step {} : {}".format(index + 1,
#                                                             accuracy))
#
#             print("\n")
#
#             total_loss = 0.0
#             saver.save(sess, 'checkpoints/HAR-Conv1LSTM2/HAR_Conv1LSTM2', index)
#
#     print("Optimization Finished")
#     print("Total time: {} seconds".format(time.time() - start_time))
