# -*- coding: utf-8 -*-
"""
IE 534 - Homework 1
Author - Pulkit Dixit, pulkitd2
"""

import numpy as np
import h5py
import time
from random import randint
import math

#load MNIST data and create training and testing sets:
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
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

#Initializing values of numbers of inputs, outputs and hidden layers:
#Number of inputs:
num_inputs = 28*28
#Number of outputs:
num_outputs = 10
#Number of nodes in the hidden layer:
dH = 200

#Defining a function to create the parameters:
def create_params(num_inputs, num_outputs, dH):
    model = {}
    np.random.seed(1)
    model['W'] = np.random.randn(dH,num_inputs) / np.sqrt(num_inputs)
    np.random.seed(2)
    model['C'] = np.random.randn(num_outputs,dH) / np.sqrt(dH)
    model['b1'] = np.zeros((dH))
    model['b2'] = np.zeros((num_outputs))
    return model

#Creating parameters:
model = create_params(num_inputs, num_outputs, dH)

#Creating a function for softmax calculation:
def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ

#Creating a function for forward propogation:
def forward(x, model):
    W = model['W']
    b1 = model['b1']
    C = model['C']
    b2 = model['b2']
    
    Z = np.dot(W, x) + b1

    H = np.maximum(0, Z)        #ReLU activation function

    U = np.dot(C, H) + b2

    p = softmax_function(U)     #softmax function
    
    return p, H, Z

#Defining cross-entropy loss as the cost function:
def cost(y, p):
    if y==0:
        return -math.log(p[0])
    elif y==1:
        return -math.log(p[1])
    elif y==2:
        return -math.log(p[2])
    elif y==3:
        return -math.log(p[3])
    elif y==4:
        return -math.log(p[4])
    elif y==5:
        return -math.log(p[5])
    elif y==6:
        return -math.log(p[6])
    elif y==7:
        return -math.log(p[7])
    elif y==8:
        return -math.log(p[8])
    elif y==9:
        return -math.log(p[9])
    
#Creating a function to calculate the derivative of the ReLU function:
def ReLU_derivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

#Creating a function for backward propogation:
def backward(x, y, model, p, H, Z, dH):
    W = model['W']
    b1 = model['b1']
    C = model['C']
    b2 = model['b2']
    
    #Creating the function e(y):
    eY = np.zeros(10)
    eY[y] = 1
    
    dpdU = -(eY - p)
    
    delta = np.dot(C.T, dpdU)
    
    #Partial derivative for b2:
    dpdb2 = dpdU
    
    #Partial derivative for C:
    dpdC = np.dot(np.reshape(dpdU, (10,1)), np.reshape(H, (dH,1)).T)
    
    #Calculating ReLU derivative:
    sigma = ReLU_derivative(Z)
    
    #Partial derivative for b1:
    dpdb1 = np.multiply(delta, sigma)
    
    #Partial derivative for W:
    dpdW = np.dot(np.reshape(np.multiply(delta, sigma), (dH,1)), np.reshape(x, (784, 1)).T)
    
    grads = {'dW': dpdW,
             'db1': dpdb1,
             'dC': dpdC,
             'db2': dpdb2}
    
    return grads

#Creating a function for stochastic gradient descent:
def sgd(learning_rate, model, grads):
    W = model['W']
    b1 = model['b1']
    C = model['C']
    b2 = model['b2']

    dW = grads['dW']
    db1 = grads['db1']
    dC = grads['dC']
    db2 = grads['db2']
    
    #Updating parameters
    W = W - learning_rate*dW
    b1 = b1 - learning_rate*db1
    C = C - learning_rate*dC
    b2 = b2 - learning_rate*db2
  
    new_model = {'W': W,
                 'C': C,
                 'b1' : b1,
                 'b2' : b2
                 }

    return new_model

#Initializing the learning rate and the number of epochs:
learning_rate = 0.01
num_epochs = 10

time1 = time.time()

#Training the model:
for epochs in range(num_epochs):
    print('Current epoch: ', epochs, '/', num_epochs)
    
    #Setting the learning rate schedule:
    if (epochs > 5):
        learning_rate = 0.001
    if (epochs > 10):
        learning_rate = 0.0001
    if (epochs > 15):
        learning_rate = 0.00001
    
    total_correct = 0
    
    #Performing forward and backward propogation on the training data:
    for n in range(len(x_train)):
        n_random = randint(0,len(x_train)-1)
        y = y_train[n_random]
        x = x_train[n_random][:]
        p, H, Z = forward(x, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        grads = backward(x, y, model, p, H, Z, dH)
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
    p, H, Z = forward(x, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print('Test accuracy: ', total_correct/np.float(len(x_test)))
























