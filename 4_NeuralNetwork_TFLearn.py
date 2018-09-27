"""
Example of a multilayer-perceptron learning XOR using TFLearn
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/tensorflow-comparison
"""

from __future__ import division, print_function, absolute_import
import numpy as np
import tflearn

###### Setup ######

## definiition of the function to be learned: xor
def xor(v):
    w = [0 if val < 0 else 1 for val in v]
    return 1 if not((w[0] + w[1] -1)%2) else -1

## possible input and output values
inputs = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
outputs = [[xor(d)] for d in inputs]

## calculation of loss while learning (just for display)
def get_current_loss(loss_op,session):
    feed = [{X: [inputs[i]],Y: [outputs[i]]} for i in range(4)]
    l = [loss_op.eval(feed_dict=feed[i],session=session) for i in range(4)]
    return np.sum(l)/4.0

## parameters
datasize = 1000     # length of dataset
dim_X = 2           # dimension of inputs
dim_Y = 1           # dimension of outputs
dim_h = 2           # dimension of hidden layer
epochs = 5          # number of epochs for training

## data generation
data = np.array([inputs[np.random.choice(range(4))] for i in range(datasize)])
labels = [[xor(d)] for d in data]

###### Neural Network ######

# input layer
X = tflearn.input_data(shape=[None, dim_X])
# hidden layer
h = tflearn.fully_connected(X, dim_h, activation='tanh')
# output layer
Y = tflearn.fully_connected(h, dim_Y, activation='tanh')

# optimizer
# sgd = tflearn.SGD(learning_rate=.01)
adam = tflearn.Adam(learning_rate=0.01, beta1=0.99)

# combining output layer with optimizer and loss
net = tflearn.regression(Y, optimizer=adam, loss='mean_square')


###### Training ######

model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(data, labels, batch_size=1, n_epoch=epochs)

print(model.predict(inputs))
