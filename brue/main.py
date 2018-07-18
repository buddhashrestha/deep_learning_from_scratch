from classifiers.fc_nns import *
from solver import *
from data_utils import *

data = load_mnist()

# data = get_CIFAR10_data()
print(data['X_train'].shape)
best_model = None
################################################################################
# TODO: Train the best FullyConnectedNet that you can on CIFAR-10. You might   #
# batch normalization and dropout useful. Store your best model in the         #
# best_model variable.                                                         #
################################################################################

learning_rate = 3.113669e-04
weight_scale = 2.461858e-02
learning_rate = 0.008
# ### CONSTANTS ###
activations = ["relu","relu","relu","relu"]
model = FullyConnectedNet([100, 100, 100, 100],activations,
                          input_dim=784,
                weight_scale=weight_scale, dtype=np.float64)
solver = Solver(model, data,
                print_every=100, num_epochs=10, batch_size=250,
                update_rule='adam',
                optim_config={
                  'learning_rate': learning_rate,
                }
         )
solver.train()

best_model = model

y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1)
print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())