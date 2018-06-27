import time
import numpy as np
import matplotlib.pyplot as plt

from propagation import *
from activations import *
from Initializations import *
from costs import *
from datas import *




def L_layer_model(X, Y, activations, layers_dims, learning_rate=0.085 , num_iterations=5000, print_cost=False): #lr was 0.009

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


    #plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

train_x,train_y,test_x,test_y = load_cifar('./Assignment One/cifar-10-python/cifar-10-batches-py')
# train_x,train_y,test_x,test_y = load_mnist("mnist.data",split=0.8)


### CONSTANTS ###
activations = ["relu","relu","relu","relu"]

layers_dims = [3072, 28, 15, 7, 10] #  5-layer model for cifar-10 data
# layers_dims = [784, 12, 7, 4, 10] #  5-layer model for mnist data

parameters = L_layer_model(train_x,train_y , activations, layers_dims, num_iterations=800, print_cost=True)

pred_train = predict(train_x, train_y, parameters,activations)

pred_test = predict(test_x, test_y, parameters,activations)

print("Training accuracy :", pred_train)

print("Test data accuracy :", pred_test)

