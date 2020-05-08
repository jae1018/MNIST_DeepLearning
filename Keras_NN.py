#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uses Keras to implement a deep neural network that mirrors the c++ implementation
we created (only this one works).

Note:
    This code snippet has been appropriated from an introductory code
    on the brevity of using Keras for machine learning algorithms. I have added
    / modified very little code, largely adding comments to provide explanation
    for the functionality of certain segments.
    
    The source for the original code is at
    https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
"""
	
# Baseline MLP for MNIST dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


# load data (both training [10K] and testing [60K] sets)
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')


# normalize inputs from 0-255 to 0-1
# originally was in grayscale (represented by ints from 0 to 255), so we normalize
X_train = X_train / 255
X_test = X_test / 255


# converts vector of labels into matrix of classes, i.e.
# for this data, labels are integers from 0 to 9 but to_categorically() creates a vector *per label*
# with each vector's number of elements = the number of possible classes (here, that's 10 b/c there are 10 integers)
# E.g. the vector [0,2,7,4] becomes [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# -- Define baseline model object --
# We will use a 3-layered neural network: 1 input, 1 hidden, and 1 output.
def baseline_model():
	# create model
    # Keras works by requiring that you either..
    # (1) instantiate your model object first and then add parameters later, as is done here
    #     (these param's will determine the function of hidden and output layers), or
    # (2) instantiate your model with all the desired parameters through the constructor
    #     (see at https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
    #     for an example))
	model = Sequential() 
    
    # -- Adding param's --
    # This first line specifies the function of the single hidden layer:
    # The input to this layer is to be a vector with num_pixels (= 784) number of elements
    # The output is a vector with 20 elements
    # Its activations function is sigmoid
    #     (can specify others: "tanh", "softmax", "relu")
    # kernal_initializer determines by what probability distribution are the weights sampled from
    #     (i.e. "random_uniform", "normal", "zeros", "constant", etc.)
    # For more detail on the Dense, see the Keras documentation at https://keras.io/layers/core/
	model.add(Dense(20, input_dim=num_pixels, kernel_initializer='random_uniform', activation='sigmoid'))
    
    # Defines function for 3rd layer (the 2nd hidden layer)
	model.add(Dense(15, input_dim=20, kernel_initializer='random_uniform', activation='sigmoid'))
    
    # This specifies the function of the output layer:
    # Its output is num_classes (= 10)
    # Its weight initialization is constructed according to a normal distribution
    # Its activation function is softmax
	model.add(Dense(num_classes, input_dim = 15, kernel_initializer='random_uniform', activation='sigmoid'))

	# -- Compile model --
    # This line specifies how the data is evaluated after the input has propagated through to the final layer:
    # The cost function here is the standard mean square error 
    # The optimizer is stochastic gradient descent (related to how do you want to iterate towards better, more accurate weights; others include
    #     "rmsprop", "adagrad", "ADAM")
    # The metrics determines how the result of the network is evaluated with respect to the answer provided in the data set
    #     (accuracy is simply number correct/total sample size, but others are available)
    # For more info on Compile, see an example at https://keras.io/getting-started/sequential-model-guide/
	model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()

# -- Fit the model --
# Trains the model for a specified number of epochs.
# The x_train and y_train vars specify the tests and their corresponding answer keys
# After training is done, the algorithm's performance is tracked against the validation set
# The batch sze determines how many tests are executed before the weights and biases are updated
#     from the gradient.
# verbose decides what info to return during training. Specifically, the documentation describes
#     it with " verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch). "
# For more info, read the docs at https://keras.rstudio.com/reference/fit.html
print("\nExecuting model.fit()...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# Final evaluation of the model
print("\nExecuting model.evaluate()...")
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))