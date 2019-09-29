# -*- coding: utf-8 -*-
"""
IE 534 - Homework 2
Author - Pulkit Dixit, pulkitd2
"""

import numpy as np
import h5py
import time
from random import randint
import math

#load MNIST data and create training and testing sets:
MNIST_data = h5py.File('C:/Users/Pulkit Dixit/Documents/MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()

print('Dimensions of training and testing variables and labels:')
print('x_train: ', x_train.shape)
print('y_train: ', y_train.shape)
print('x_test: ', x_test.shape)
print('y_test: ', y_test.shape)

#Initializing parameters:
d = 28
kx = 3 #number of rows in the filter
ky = 3 #number of columns in the filter
ch = 5 #number of channels
num_outputs = 10

#Defining a function to create the parameters:
def create_params(d, kx, ky, num_outputs, ch):
    model = {}
    np.random.seed(1)
    model['W'] = np.random.randn(num_outputs, d-ky+1, d-kx+1, ch)/np.sqrt(num_outputs*(d-ky+1)*(d-kx+1))
    model['b'] = np.zeros((num_outputs))
    model['K'] = np.random.randn(kx, ky, ch)/np.sqrt(kx*ky)
    return model

def conv(X, K, d, kx, ky, ch):
    Z = np.zeros((d-ky+1, d-kx+1, ch))
    for i in range(d-ky+1):
        for j in range(d-kx+1):
            for ch_num in range(ch):
                Z[i,j,ch_num] = np.sum(np.multiply(x[i:i+K.shape[0], j:j+K.shape[1]], K.T))
    return Z

#Creating a function for softmax calculation:
def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ

def forward(x, model, d, kx, ky, ch):
    W = model['W']
    b = model['b']
    K = model['K']
    
    Z = conv(x, K, d, kx, ky, ch)

    H = np.maximum(0, Z)        #ReLU activation function
        
    U = np.tensordot(W, H, axes = ([3,2,1], [2,1,0])) + b
    
    p = softmax_function(U)     #softmax function
    
    return p, Z, H

#Creating a function to calculate the derivative of the ReLU function:
def ReLU_derivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

#Creating a function for backward propogation:
def backward(x, y, model, p, H, Z):
    W = model['W']
    b = model['b']
    K = model['K']
    
    #Creating the function e(y):
    eY = np.zeros(10)
    eY[y] = 1
    
    dpdU = -(eY - p)
    
    #Partial derivative for b:
    dpdb = dpdU
    
    #Partial derivative for W:
    dpdW = np.zeros((W.shape[0], W.shape[1], W.shape[2], W.shape[3]))
    for i in range(W.shape[0]):
        dpdW[i,:,:,:] = dpdU[i] * H
    
    delta = np.tensordot(W, dpdU, axes = ([0], [0]))
    sigma = ReLU_derivative(Z)
    
    #Partial derivative for K:
    mul = np.multiply(delta, sigma)
    dpdK = np.zeros((model['K'].shape[0], model['K'].shape[1], model['K'].shape[2]))
    for i in range(ky):
        for j in range(kx):
            for ch_num in range(ch):
                dpdK[i,j,ch_num] = np.sum(np.multiply(x[i:i+mul.shape[0], j:j+mul.shape[1]], mul.T))
                
    grads = {'dW': dpdW,
             'db': dpdb,
             'dK': dpdK
             }
    
    return grads

#Defining a function for stochastic gradient descent:
def sgd(learning_rate, model, grads):
    W = model['W']
    b = model['b']
    K = model['K']

    dW = grads['dW']
    db = grads['db']
    dK = grads['dK']
    
    #Updating parameters
    W = W - learning_rate*dW
    b = b - learning_rate*db
    K = K - learning_rate*dK
  
    new_model = {'W': W,
                 'K': K,
                 'b' : b
                 }

    return new_model

learning_rate = 0.01
num_epochs = 10

model = create_params(d, kx, ky, num_outputs, ch)

time1 = time.time()

#Training the model:
for epochs in range(num_epochs):
    print('Current epoch: ', epochs+1, '/', num_epochs)
    
    #Setting the learning rate schedule:
    if (epochs > 2):
        learning_rate = 0.001
    if (epochs > 5):
        learning_rate = 0.0001
    if (epochs > 7):
        learning_rate = 0.00001
#    if (epochs > 10):
#        learning_rate = 0.000001
#    if (epochs > 12):
#        learning_rate = 0.000001
    
    total_correct = 0
    
    #Performing forward and backward propogation on the training data:
    for n in range(len(x_train)):
        #n_random = randint(0,len(x_train)-1)
        n_random = randint(0,10000)
        y = y_train[n_random]
        x = x_train[n_random][:]
        x = np.reshape(x,(-1,28))
        p, Z, H = forward(x, model, d, kx, ky, ch)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        grads = backward(x, y, model, p, H, Z)
        model = sgd(learning_rate, model, grads)
    
    #Printing the accuracy:
    print(total_correct/np.float(len(x_train)))

time2 = time.time()
print('Time taken for training: ', time2-time1)
print('Training accuracy: ', total_correct/np.float(len(x_train)))

#Testing the test data:
total_correct = 0
for n in range(len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    x = np.reshape(x,(-1,28))
    p, H, Z = forward(x, model, d, kx, ky, ch)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print('Test accuracy: ', total_correct/np.float(len(x_test)))























