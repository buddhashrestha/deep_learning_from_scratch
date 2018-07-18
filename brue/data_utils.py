from __future__ import print_function

from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir =  '/home/buddha/projects/CS231n/assignment1/cs231n/datasets/cifar-10-python/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }



def load_mnist(num_training=50000, num_validation=5000, num_test=5000):
    mnist_dir = "../mnist.data"
    mnist =  pickle.load(open(mnist_dir, "rb"))
    print(mnist.data.shape)
    # num_training = int(mnist.data.shape[0] * split)
    # num_test = int(mnist.data.shape[0] * (.5 * (1 - split)))
    # num_validation = int(mnist.data.shape[0] * (.5 * (1 - split)))
    mask_training = list(range(num_training, num_training + num_validation))
    mask_validation = list(range(num_training, num_training + num_validation))
    mask_test = list(range(num_training + num_validation, num_training + num_validation + num_test))

    X_train = mnist.data[mask_training]
    y_train = mnist.target[mask_training]

    X_val = mnist.data[mask_validation]
    y_val = mnist.target[mask_validation]

    X_test = mnist.data[mask_test]
    y_test = mnist.target[mask_test]


    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }


# def split_test_train(data_name, percentage=0.8):
#
#     train_x = data_name.data[:training_samples].T  # train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
#     train_y = np.array([data_name.target[:training_samples]])
#     test_x = data_name.data[training_samples:].T
#     test_y = np.array([data_name.target[training_samples:]])  # test_x_orig.reshape(test_x_orig.shape[0], -1).T
#     return train_x,train_y,test_x,test_y

def whiten_data(x):
        return x/255
