# MNIST_DeepLearning

This program uses Deep Learning to perform digit detection on hand-drawn images of numbers (the classic MNIST data set). After the program has been trained on the training data, the driver program DNN_Main.cxx can be slightly modified so that the algorithm can perform against the validation set.

Neural Network Implementation:
We use a multi-hidden layer network to read the handwritten images of the MNIST data set. The number of neurons and layer used can be variable and is determined in the header file DNN.h (so long as the input layer has 784 nuerons, the output layer has 10, and that there are at least 2 hidden layers). Our network is very basic and uses some of the simpler options available for its functions: the cost function is mean square error, the activation function is sigmoid, and the optimizer is stochastic gradient descent with a batch size of 30 (but this too can be varied in the header file).
Every 10K tests, the program saves its weights and biases to .csv files in the build directory. If these files are not detected at startup, the program will build its own weights and biases from scratch.

Input Data Handling:
Using 3rd party MIT-Licensed software, we are able to convert the binaries of the original testing files into a format appropriate for the network to read. This software can be found at https://github.com/wichtounet/mnist.

Bugs and Issues:
This program fails to adapt over randomized testing. However, providing the program batches of repeated, consecutive tests enables it to learn and causes its accuracy on the validation suite to become functional at values of ~60% (including randomizing the tests in the validation set)
