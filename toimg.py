import tensorflow as tf
import numpy as np
from collections import OrderedDict

##################################################################################
# Layer
##################################################################################

# pad = ceil[ (kernel - stride) / 2 ]

def get_weight(weight_shape, gain, lrmul):
    fan_in = np.prod(weight_shape[:-1])  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in)  # He init

    # equalized learning rate
    init_std = 1.0 / lrmul
    runtime_coef = he_std * lrmul

    # create variable.
    weight = tf.get_variable('weight', shape=weight_shape, dtype=tf.float32,
                             initializer=tf.initializers.random_normal(0, init_std)) * runtime_coef
    return weight

def conv(x, channels, kernel=3, stride=1, gain=np.sqrt(2), lrmul=1.0, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        weight_shape = [kernel, kernel, x.get_shape().as_list()[-1], channels]
        weight = get_weight(weight_shape, gain, lrmul)

        x = tf.nn.conv2d(input=x, filter=weight, strides=[1, stride, stride, 1], padding='SAME')

        return x


def torgb(x, res, sn=False):
    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('ToRGB'):
            x = conv(x, channels=3, kernel=1, stride=1, gain=1.0, lrmul=1.0, sn=sn)
            x = apply_bias(x, lrmul=1.0)
    return x


def toimg(x, res, sn=False):
    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('ToRGB'):
            x = conv(x, channels=1, kernel=1, stride=1, gain=1.0, lrmul=1.0, sn=sn)
            x = apply_bias(x, lrmul=1.0)
    return x



def apply_noise(x):
    with tf.variable_scope('Noise'):
        noise = tf.random_normal([tf.shape(x)[0], x.shape[1], x.shape[2], 1])
        weight = tf.get_variable('weight', shape=[x.get_shape().as_list()[-1]], initializer=tf.initializers.zeros())
        weight = tf.reshape(weight, [1, 1, 1, -1])
        x = x + noise * weight

    return x

def apply_bias(x, lrmul):
    b = tf.get_variable('bias', shape=[x.shape[-1]], initializer=tf.initializers.zeros()) * lrmul

    if len(x.shape) == 2:
        x = x + b
    else:
        x = x + tf.reshape(b, [1, 1, 1, -1])

    return x
