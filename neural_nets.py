
import matplotlib.pyplot as plt

from propagation import *
from activations import *
from Initializations import *
from costs import *
from datas import *
from optimizers import *



def L_layer_model(X, Y, activations, layers_dims, learning_rate=0.085 , num_iterations=5000, print_cost=False): #lr was 0.009

    np.random.seed(1)
    costs = []                         # keep track of cost
    threshold = 0.0001

    # Parameters initialization.
    parameters = initialize(activations,layers_dims)
    # v = initialize_velocity(parameters)
    v, s = initialize_adam(parameters)
    Y = one_hot_encoding(Y)
    Y = Y.reshape(Y.shape[0],X.shape[1])
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters,activations)

        # Compute cost and derivative with respect to output
        cost, dAL  = compute_cost(AL,Y,"cross_entropy")

        # Backward propagation.
        grads = L_model_backward(dAL, caches,parameters,activations)

        # Update parameters.
        # parameters = update_parameters(parameters, grads, learning_rate)
        # parameters,v  = update_parameters_with_momentum(parameters, grads, v, learning_rate, beta=0.9)
        parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t=2)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)


    #plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

    return parameters


def model(X, Y,activations, layers_dims, optimizer, learning_rate=0.07, mini_batch_size=2048, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
    """
    3-layer neural network model which can be run in different optimizer modes.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- python dictionary containing your updated parameters
    """

    L = len(layers_dims)  # number of layers in the neural networks
    costs = []  # to keep track of the cost
    t = 0  # initializing the counter required for Adam update
    seed = 10  # For grading purposes, so that your "random" minibatches are the same as ours

    # Initialize parameters
    parameters = initialize(activations, layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # Optimization loop
    for i in range(num_epochs):

        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = L_model_forward(X, parameters, activations)

            # Compute cost and derivative with respect to output
            cost, dAL = compute_cost(AL, Y, "cross_entropy")

            # Backward propagation.
            grads = L_model_backward(dAL, caches, parameters, activations)

            # Update parameters.
            # parameters = update_parameters(parameters, grads, learning_rate)
            # parameters,v  = update_parameters_with_momentum(parameters, grads, v, learning_rate, beta=0.9)
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t=2)

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1  # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2, epsilon)

        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    # plt.plot(costs)
    # plt.ylabel('cost')
    # plt.xlabel('epochs (per 100)')
    # plt.title("Learning rate = " + str(learning_rate))
    # plt.show()

    return parameters


def predict(X, y, parameters,activations):

    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m), dtype=int)
    # Forward propagation
    probas, caches = L_model_forward(X, parameters,activations)

    predicted_output = np.argmax(probas, axis=0)

    y = y.reshape(predicted_output.shape)

    correct_labels = 0

    number_of_samples = len(predicted_output)

    for i in range(0, number_of_samples):
        if (predicted_output[i] == y[i]):
            correct_labels += 1

    return correct_labels / number_of_samples
