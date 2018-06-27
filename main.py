from datas import *
from neural_nets import *


################################## FOR CIFAR-10 DATA SET ###################################################

train_x,train_y,test_x,test_y = load_cifar('./Assignment One/cifar-10-python/cifar-10-batches-py')


### CONSTANTS ###
activations = ["tanh","relu","tanh","relu"]

layers_dims = [3072, 28, 15, 7, 10] #  5-layer model for cifar-10 data

parameters = L_layer_model(train_x,train_y , activations, layers_dims, num_iterations=800, print_cost=True)

pred_train = predict(train_x, train_y, parameters,activations)

pred_test = predict(test_x, test_y, parameters,activations)

print("Training accuracy :", pred_train)

print("Test data accuracy :", pred_test)




################################ FOR MNIST DATA SET ####################################################

train_x,train_y,test_x,test_y = load_mnist("mnist.data",split=0.8)


### CONSTANTS ###
activations = ["tanh","relu","tanh","relu"]

layers_dims = [784, 12, 7, 4, 10] #  5-layer model for mnist data

parameters = L_layer_model(train_x,train_y , activations, layers_dims, num_iterations=2500, print_cost=True)

pred_train = predict(train_x, train_y, parameters,activations)

pred_test = predict(test_x, test_y, parameters,activations)

print("Training accuracy :", pred_train)

print("Test data accuracy :", pred_test)
