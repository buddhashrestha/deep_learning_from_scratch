import time
import numpy as np
#import matplotlib.pyplot as plt

from dnn_app_utils import *
from activations import *

import cloudpickle as pickle
mnist23 = pickle.load( open( "mnist.data", "rb" ) )


np.random.seed(1)


# Example of a picture
index = 10


training_samples = 60000


# Reshape the training and test examples 
train_x_flatten = mnist23.data[:training_samples].T  #train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
train_y = np.array([mnist23.target[:training_samples]])
test_x_flatten = mnist23.data[training_samples:].T
test_y = np.array([mnist23.target[training_samples:]]) #test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.
train_y = train_y
test_y = test_y
print(train_y)
print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))


### CONSTANTS ###
layers_dims = [784, 28, 15, 7, 10] #  5-layer model
# layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
# GRADED FUNCTION: n_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate=0.09, num_iterations=5000, print_cost=False): #lr was 0.009

    np.random.seed(1)
    costs = []                         # keep track of cost
    threshold = 0.0001
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    Y = one_hot_encoding(Y)
    Y = Y.reshape(Y.shape[0],training_samples)
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        CE,cache = softmax(AL)

        # Compute cost.
        cost = cross_entropy_cost(CE, Y)

        # Backward propagation.
        grads = L_model_backward(CE, AL, Y, caches)
        ### END CODE HERE ###
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)
            
    # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

    return parameters

parameters = L_layer_model(train_x,train_y , layers_dims, num_iterations=3000, print_cost=True)

pred_train = predict(train_x, train_y, parameters)

pred_train = predict(test_x, test_y, parameters)
print("Everything good :", pred_train)

