"""
Example of a multilayer-perceptron learning XOR using numpy
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/tensorflow-comparison
"""

import numpy as np

###### Setup ######

## definition of the function to be learned: xor
def xor(v):
    w = [0 if val < 0 else 1 for val in v]
    return 1 if not((w[0] + w[1] -1)%2) else -1

## possible input and output values
inputs = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
# inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
outputs = [xor(d) for d in inputs]

## calculation of loss while learning (just for display)
def get_current_loss():
    l = [(net(x)-y)**2 for x,y in zip(inputs,outputs)]
    return np.sum(l)/4.0

## general parameters
datasize = 1000     # length of dataset
epochs = 5          # number of epochs for training
# lr = .001            # learning rate

## data generation
data = np.array([inputs[np.random.choice(range(4))] for i in range(datasize)])
labels = [xor(d) for d in data]

###### Neural Network ######

## weights and biases with hard-coded dimensions for simplicity
weights = {
    'h': np.random.normal(size=[2,2]),
    'y': np.random.normal(size=[2]),
    }
biases = {
    'h': np.random.normal(size=[2]),
    'y': np.random.normal(size=[1]),
    }

## hidden layer
def h(x): return np.tanh(np.einsum('ij,j->i',weights['h'],x) + biases['h'])
## output layer
def out(hidden): return np.tanh(np.sum(weights['y']*hidden) + biases['y'])
## full neural network
def nn(x):return out(h(x))

## gradient descent
def gradientDescent(x,y,theta,grad_theta,lr):
    return theta - lr * grad_theta(x,y)

## gradients of loss with respect to weights and biases
signals = {}
grad_weights = {}
grad_biases = {}
signals['y'] = lambda x,y : (nn(x)-y)*(1-nn(x)**2)
signals['h'] = lambda x,y : signals['y'](x,y)*weights['y']*(1-h(x)**2)
grad_weights['y'] = lambda x,y : signals['y'](x,y)*h(x)
grad_biases['y'] = lambda x,y : signals['y'](x,y)
grad_weights['h'] = lambda x,y : np.einsum('i,j->ij',signals['h'](x,y),x)
grad_biases['h'] = lambda x,y : signals['h'](x,y)

loss=[]
for epoch in range(epochs):
    for i in range(len(data)):
        x,y = data[i],labels[i]
        ## update weights and biases
        weights['h'] = gradientDescent(x,y,weights['h'],grad_weights['h'],lr=0.01)
        weights['y'] = gradientDescent(x,y,weights['y'],grad_weights['y'],lr=0.01)
        biases['h'] = gradientDescent(x,y,biases['h'],grad_biases['h'],lr=0.01)
        biases['y'] = gradientDescent(x,y,biases['y'],grad_biases['y'],lr=0.01)
        loss.append((nn(x)-y)**2)
    #print(get_current_loss())

print('x->y:\n {}'.format([net(x) for x in inputs]))
print('x->y:\n {}'.format([h(x) for x in inputs]))

import matplotlib.pyplot as plt
plt.plot(range(len(loss)),loss)
plt.show()
