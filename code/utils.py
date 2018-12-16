import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)

    def f1(): return 0.5 * tf.square(residual)

    def f2(): return delta * residual - 0.5 * tf.square(delta)

    return tf.cond(residual < delta, f1, f2)


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def get_batch(X, y, batch_size, index):
    batch_flag = 0
    assert X.shape[0] == y.shape[0]
    n_batches = int(X.shape[0] / batch_size)

    return X[index * batch_size:(index + 1) * batch_size], y[index * batch_size:(index + 1) * batch_size]


"****************************************************************************************"
"********************************** 1. Load Dataset *****************************************"
"****************************************************************************************"

import numpy as np
import scipy.io
import pickle as cp

default_opportunity_dataset_path = "/Users/chentailin/Desktop/workspace/paper_multimodality/Opportunity/data/activity-dataset (113_wearable+40_ambient).data"


# Function to load the preprocessed Activity dataset
# 113 wearable sensors + 40 ambient sensors
# actually, we only use the wearable sensor
def load_dataset(dataset_name, dataset_file_path=default_opportunity_dataset_path):
    """

    :param dataset_name: 'opportunity' or 'pamap2' dataset
    :param filename: the path of the dataset
    :return: x_train,
             y_train,
             x_test,
             y_test
    """

    if dataset_name == 'opportunity':
        file = open(dataset_file_path, 'rb')
        data = cp.load(file)
        file.close()

        x_train, y_train = data[0]
        x_test, y_test = data[1]

        print(" ..from file {}".format(dataset_file_path))
        print(" ..reading instances: x_train {0}, x_test {1}".format(x_train.shape, x_test.shape))
        print(" ..reading instances: y_train {0}, y_test {1}".format(y_train.shape, y_test.shape))

        return x_train, y_train, x_test, y_test
    if dataset_name == 'pamap2':
        print(" ..from file {}".format(dataset_file_path))
        pass

    else:
        print("Please chose the right dataset")


"****************************************************************************************"
"********************************** 2. Sliding Window *****************************************"
"****************************************************************************************"

import numpy as np
from numpy.lib.stride_tricks import as_strided as ast  # a very powerful method for 分块矩阵


def norm_shape(shape):
    '''
        Normalize numpy array shapes so they're always expressed as a tuple,
        even for one-dimensional shapes.

        Parameters
            shape - an int, or a tuple of ints

        Returns
            a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        pass
    raise TypeError("Shape must be an int, or a tuple of ints")


def sliding_window(a, ws, ss=None, flatten=True,shape_flag=False):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError( \
            'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError( \
            'ws cannot be larger than a in any dimension.\
     a.shape was %s and ws was %s' % (str(a.shape), str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a, shape=newshape, strides=newstrides)
    if not flatten:
        if shape_flag:
            strided = np.squeeze(strided,axis=1)
            return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i: i != 1, dim)
    return strided.reshape(dim)


def op_sliding_window(x, y, ws, ss):
    """
    This is used for preprocessing the data into a desired format
    Inputs are 2-D and outputs are 3-D.
    For example:
                (9,9) --> (3,3,9)

    :param data_x: raw wearabel accelerameter data
    :param data_y: label for each time steps
    :param ws: window_length
    :param ss: slidng_stride
    :return: the processed data  X and label y
    """

    data_x = sliding_window(a=x, ws=(ws, x.shape[1]), ss=(ss, 1), flatten=False)
    #     data_x = sliding_window(data_x,ws,ss)

    data_y = np.asarray([[i[-1]] for i in sliding_window(y, ws, ss, flatten=False)])

    print("Sliding Window Successfully ")
    print("Input shape is {0} and {1},\nOutput shape is {2} and {3}.".format(x.shape, data_x.shape, y.shape,
                                                                             data_y.shape))
    return data_x.reshape(-1, ws, data_x.shape[3]).astype(np.float32), data_y.reshape(-1).astype(np.uint8)


def input_segmentation(input, segmentation_size=32, segmentation_stride=16, random_seed=1):
    """

    :param input:
    :param segmentation_size:
    :param segmentation_stride:
    :param random_seed:
    :return: a list : store the processed input
    """
    batch_size = input.shape[0]
    window_length = input.shape[1]
    num_dim = input.shape[2]

    results = []

    for i in range(batch_size):
        temp_input = input[i]
        new_output = sliding_window(a=temp_input,ws=(segmentation_size,temp_input.shape[1]),ss=(segmentation_stride,1),flatten=False,shape_flag=True)
        results.append(new_output)
    return results


"****************************************************************************************"
"********************************** 3. One-Hot Encoder *****************************************"
"****************************************************************************************"

from keras.utils import to_categorical


def one_hot_encoder(y, num_classes):
    y_one_hot = to_categorical(y=y, num_classes=num_classes)
