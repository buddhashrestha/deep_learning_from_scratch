from classifiers.activation_nets import *
from solver import *
from data_utils import *

# data = load_mnist()

data = get_CIFAR10_data()
print(data['X_train'].shape)
best_model = None
################################################################################
# TODO: Train the best FullyConnectedNet that you can on CIFAR-10. You might   #
# batch normalization and dropout useful. Store your best model in the         #
# best_model variable.                                                         #
################################################################################

learning_rate = 3.113669e-04
weight_scale = 2.461858e-02
# learning_rate = 0.008
# ### CONSTANTS ###
activations = ["relu","relu","relu","relu"]

best_val = -1
learning_rates = [1e-2, 1e-3,3.113669e-04]
regularization_strengths = [0.04, 0.05, 0.06]

results = {}
iters = 2000  # 100
# for lr in learning_rates:
#     for rs in regularization_strengths:
#         model = ActivationNet([100, 100, 100, 100], activations,
#                           input_dim=3072,
#                           dropout=0.0, use_batchnorm=False, reg=rs,
#                           weight_scale=weight_scale, dtype=np.float64)
#         solver = Solver(model, data,
#                         print_every=1000, num_epochs=10, batch_size=250,
#                         update_rule='adam',
#                         optim_config={
#                             'learning_rate': learning_rate,
#                         }
#                         )
#         acc_train, acc_val = solver.train()
#
#         results[(lr, rs)] = (acc_train, acc_val)
#
#         if best_val < acc_val:
#             best_val = acc_val
#             best_model = model
# print('best validation accuracy achieved during cross-validation: %f' % best_val)
#
# # Print out results.
# for lr, reg in sorted(results):
#     train_accuracy, val_accuracy = results[(lr, reg)]
#     print( 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
#                 lr, reg, train_accuracy, val_accuracy))

model = ActivationNet([100, 100, 100, 100],activations,
                          input_dim=3072,
                            dropout=0.05, use_batchnorm=False, reg=0.05,
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