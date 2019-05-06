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

## general parameters
datasize = 100      # length of dataset
dim_X = 2           # dimension of inputs
dim_Y = 1           # dimension of outputs
dim_h = 2           # dimension of hidden layer
episodes = 5000     # number of episodes for training
lr = .01            # learning rate    

## data generation
data = np.array([inputs[np.random.choice(range(4))] for i in range(datasize)])
labels = [[xor(d)] for d in data]


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

## objective/loss
obj = tf.reduce_mean((Y-model)**2)

## loss and optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=.9, beta2=.999,epsilon=1e-08,name='Adam')
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(obj)


###### Training ######

## initializer
init = tf.global_variables_initializer()

## running the TF session
with tf.Session() as sess:
    sess.run(init)

    for n in range(0,episodes):
        
        sess.run(train_op, feed_dict={X:data, Y:labels})
        if n % 100 == 0:
            print('Loss for episode {}: {}'.format(n,obj.eval(session=sess,feed_dict={X:inputs, Y:outputs})))

    ## predictions of each layer:
    print('x->y:\n {}'.format(sess.run(model, feed_dict={X: inputs})))
    print('x->h:\n {}'.format(sess.run(h, feed_dict={X: inputs})))
