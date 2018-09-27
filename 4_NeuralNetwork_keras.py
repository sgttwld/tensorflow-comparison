"""
Example of a multilayer-perceptron learning XOR using Keras
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/tensorflow-comparison
"""

from __future__ import print_function
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation
from keras import optimizers


###### Setup ######

## definition of the function to be learned: xor
def xor(w):
    return 1 if not((w[0] + w[1] -1)%2) else -1

## possible input and output values
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
outputs = [[xor(d)] for d in inputs]

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

## input layer
X = Input(shape=(dim_X,))
## hidden layer
h = Dense(dim_h, activation='tanh')(X)
## output layer
Y = Dense(dim_Y, activation='tanh')(h)
## define full model
model = Model(inputs=X, outputs=Y)

## optimizer and loss
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)


###### Training ######

model.fit(data, labels, epochs=epochs, batch_size=1, verbose=1)

## predictions
X = np.array([[0,0],[0,1],[1,0],[1,1]])
print(model.predict(X))
