"""
Example of a multilayer-perceptron learning XOR using raw tensorflow
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/tensorflow-comparison
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf

###### Setup ######

## definition of the function to be learned: xor
def xor(w):
    return 1 if not((w[0] + w[1] -1)%2) else -1

## possible input and output values
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
outputs = [[xor(d)] for d in inputs]

## calculation of loss while learning (just for display)
def get_current_loss(loss_op,session):
    feed = [{X: [inputs[i]],Y: [outputs[i]]} for i in range(4)]
    l = [loss_op.eval(feed_dict=feed[i],session=session) for i in range(4)]
    return np.sum(l)/4.0

## general parameters
datasize = 1000     # length of dataset
dim_X = 2           # dimension of inputs
dim_Y = 1           # dimension of outputs
dim_h = 2           # dimension of hidden layer
epochs = 5          # number of epochs for training

## data generation
data = np.array([inputs[np.random.choice(range(4))] for i in range(datasize)])
labels = [xor(d) for d in data]

###### Neural Network ######

## Definition of tensor placeholders for the data and labels
X = tf.placeholder("float", [None, dim_X])
Y = tf.placeholder("float", [None, dim_Y])
## Definition of weights and biases as tensorflow variables
weights = {
    'h': tf.Variable(tf.random_normal([dim_X, dim_h])),
    'y': tf.Variable(tf.random_normal([dim_h, dim_Y])),
    }
biases = {
    'h': tf.Variable(tf.random_normal([dim_h])),
    'y': tf.Variable(tf.random_normal([dim_Y])),
    }
## hidden layer
h = tf.tanh(tf.add(tf.matmul(X, weights['h']), biases['h']))
## output layer
model = tf.tanh(tf.matmul(h, weights['y']) + biases['y'])

## loss and optimizer
loss_op = tf.losses.mean_squared_error(labels=Y,predictions=model)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss_op)

###### Training ######

## initializer
init = tf.global_variables_initializer()

## running the TF session
with tf.Session() as sess:
    sess.run(init)
    for n in range(0,epochs):
        for i in range(0,len(data)):
            # here, we use single datapoints as baches:
            batch_x, batch_y = [data[i]], np.array([labels[i]])[:, np.newaxis]
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        print('Loss for episode {}: {}'.format(n,get_current_loss(loss_op,sess)))

    ## predictions of each layer:
    print('x->y:\n {}'.format(sess.run(model, feed_dict={X: inputs})))
    print('x->h:\n {}'.format(sess.run(h, feed_dict={X: inputs})))
