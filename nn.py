import time
import numpy as np
#import matplotlib.pyplot as plt

from propagation import *
from activations import *
from Initializations import *
from Costs import *
import cloudpickle as pickle


def split_test_train(data_name, percentage=0.8):
    training_samples = int(data_name.data.shape[0] * percentage)
    train_x = data_name.data[:training_samples].T  # train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    train_y = np.array([data_name.target[:training_samples]])
    test_x = data_name.data[training_samples:].T
    test_y = np.array([data_name.target[training_samples:]])  # test_x_orig.reshape(test_x_orig.shape[0], -1).T
    return train_x,train_y,test_x,test_y

def whiten_data(x):
    return x/255

### CONSTANTS ###
activations = ["relu","relu","relu","relu"]
layers_dims = [784, 28, 15, 7, 10] #  5-layer model

def L_layer_model(X, Y, activations, layers_dims, learning_rate=0.06 , num_iterations=5000, print_cost=False): #lr was 0.009

    np.random.seed(1)
    costs = []                         # keep track of cost
    threshold = 0.0001

    # Parameters initialization.
    parameters = initialize(activations,layers_dims)
    Y = one_hot_encoding(Y)
    Y = Y.reshape(Y.shape[0],X.shape[1])
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters,activations)

        # Compute cost and derivative with respect to output
        cost, dAL  = compute_cost(AL,Y,"cross_entropy")

        # Backward propagation.
        grads = L_model_backward(AL, dAL, Y, caches,activations)

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

mnist = pickle.load(open("mnist.data", "rb"))

train_x,train_y,test_x,test_y = split_test_train(mnist,0.8)

train_x = whiten_data(train_x)

test_x = whiten_data(test_x)

parameters = L_layer_model(train_x,train_y , activations, layers_dims, num_iterations=5000, print_cost=True)

pred_train = predict(train_x, train_y, parameters)

pred_train = predict(test_x, test_y, parameters)
print("Everything good :", pred_train)

