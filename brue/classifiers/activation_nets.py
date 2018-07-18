from propagation import *
from loss import *
from initializations import *

class ActivationNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, activations, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.activations = activations
        self.dtype = dtype
        self.params = {}
        params1 = {}
        input_dims = input_dim
        # # Initialise toutput_dimshe weights and biases for each fully connected layer connected to a Relu.
        for i in range(self.num_layers - 1):
            self.params['W' + str(i+1)] = he_weight_initializer(input_dims,hidden_dims[i])
            self.params['b' + str(i+1)] = np.zeros([hidden_dims[i]])

            if self.use_batchnorm:
                self.params['beta' + str(i+1)] = np.zeros([hidden_dims[i]])
                self.params['gamma' + str(i+1)] = np.ones([hidden_dims[i]])

            input_dims = hidden_dims[i]  # Set the input dim of next layer to be output dim of current layer.

        #Initialise the weights and biases for final FC layer
        self.params['W' + str(self.num_layers)] = he_weight_initializer(input_dims,num_classes)
        self.params['b' + str(self.num_layers)] = np.zeros([num_classes])
        #
        # # Initialise the weights and biases for each fully connected layer connected to a Relu.
        # for i in range(self.num_layers - 1):
        #     self.params['W' + str(i + 1)] = np.random.normal(0, weight_scale, [input_dim, hidden_dims[i]])
        #     self.params['b' + str(i + 1)] = np.zeros([hidden_dims[i]])
        #
        #     if self.use_batchnorm:
        #         self.params['beta' + str(i + 1)] = np.zeros([hidden_dims[i]])
        #         self.params['gamma' + str(i + 1)] = np.ones([hidden_dims[i]])
        #
        #     input_dim = hidden_dims[i]  # Set the input dim of next layer to be output dim of current layer.
        #
        # # Initialise the weights and biases for final FC layer
        # self.params['W' + str(self.num_layers)] = np.random.normal(0, weight_scale, [input_dim, num_classes])
        # self.params['b' + str(self.num_layers)] = np.zeros([num_classes])

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None


        fc_cache = {}
        actv_cache={}
        bn_cache = {}
        dropout_cache = {}
        batch_size = X.shape[0]

        X = np.reshape(X, [batch_size, -1])  # Flatten our input images.

        # Do as many Affine-Activation forward passes as required (num_layers - 1).
        # Apply batch norm and dropout as required.
        for i in range(self.num_layers-1):

            fc_act, fc_cache[str(i+1)] = affine_forward(X, self.params['W'+str(i+1)], self.params['b'+str(i+1)])
            if self.use_batchnorm:
                bn_act, bn_cache[str(i+1)] = batchnorm_forward(fc_act, self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], self.bn_params[i])
                actv, actv_cache[str(i+1)] = activation_forward(bn_act,self.activations[i])
            else:
                actv, actv_cache[str(i+1)] = activation_forward(fc_act,self.activations[i])
            if self.use_dropout:
                actv, dropout_cache[str(i+1)] = dropout_forward(actv, self.dropout_param)

            X = actv.copy()  # Result of one pass through the affine-relu block.

        # Final output layer is FC layer with no relu.
        scores, final_cache = affine_forward(X, self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        # Calculate score loss and add reg. loss for last FC layer.
        loss, dsoft = softmax_loss(scores, y)
        loss += 0.5*self.reg*(np.sum(np.square(self.params['W'+str(self.num_layers)])))

        # Backprop dsoft to the last FC layer to calculate gradients.
        dx_last, dw_last, db_last = affine_backward(dsoft, final_cache)

        # Store gradients of the last FC layer
        grads['W'+str(self.num_layers)] = dw_last + self.reg*self.params['W'+str(self.num_layers)]
        grads['b'+str(self.num_layers)] = db_last

        # Iteratively backprop through each Relu & FC layer to calculate gradients.
        # Go through batchnorm and dropout layers if needed.
        for i in range(self.num_layers-1, 0, -1):

            if self.use_dropout:
                dx_last = dropout_backward(dx_last, dropout_cache[str(i)])
            dz_actv = activation_backward(dx_last, actv_cache[str(i)],self.activations[i-1])

            if self.use_batchnorm:
                dbatchnorm, dgamma, dbeta = batchnorm_backward(dz_actv, bn_cache[str(i)])
                dx_last, dw_last, db_last = affine_backward(dbatchnorm, fc_cache[str(i)])
                grads['beta' + str(i)] = dbeta
                grads['gamma' + str(i)] = dgamma
            else:
                dx_last, dw_last, db_last = affine_backward(dz_actv, fc_cache[str(i)])

            # Store gradients.
            grads['W' + str(i)] = dw_last + self.reg * self.params['W' + str(i)]
            grads['b' + str(i)] = db_last

            # Add reg. loss for each other FC layer.
            loss += 0.5 * self.reg * (np.sum(np.square(self.params['W' + str(i)])))

        return loss, grads