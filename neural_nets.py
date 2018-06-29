
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
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

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
