import _pickle as pickle


import numpy as np
import os


def get_CIFAR10_data(cifar10_dir, num_training=49000, num_validation=1000, num_test=1000):
    '''
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the neural net classifier.
    '''
    # Load the raw CIFAR-10 data
    X_train, y_train, X_test, y_test = load(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    X_train = X_train.astype(np.float64)
    X_val = X_val.astype(np.float64)
    X_test = X_test.astype(np.float64)

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2)
    X_val = X_val.transpose(0, 3, 1, 2)
    X_test = X_test.transpose(0, 3, 1, 2)
    mean_image = np.mean(X_train, axis=0)
    std = np.std(X_train)

    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    X_train /= std
    X_val /= std
    X_test /= std

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'mean': mean_image, 'std': std
    }


def load_CIFAR_batch(filename):
    ''' load single batch of cifar '''
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding ='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        Y = np.array(Y)
        return X, Y


def load_cifar(path):
    ''' load all of cifar '''
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(path, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(path, 'test_batch'))
    return Xtr.reshape(-1,Xtr.shape[0]), Ytr, \
           Xte.reshape(-1,Xte.shape[0]), Yte

def load_mnist(path,split=0.8):
    mnist =  pickle.load(open(path, "rb"))
    train_x, train_y, test_x, test_y = split_test_train(mnist, split)
    train_x = whiten_data(train_x)
    test_x = whiten_data(test_x)
    return train_x,train_y,test_x,test_y

def split_test_train(data_name, percentage=0.8):
    training_samples = int(data_name.data.shape[0] * percentage)
    train_x = data_name.data[:training_samples].T  # train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    train_y = np.array([data_name.target[:training_samples]])
    test_x = data_name.data[training_samples:].T
    test_y = np.array([data_name.target[training_samples:]])  # test_x_orig.reshape(test_x_orig.shape[0], -1).T
    return train_x,train_y,test_x,test_y

def whiten_data(x):
    return x/255
