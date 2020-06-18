import tensorflow as tf
import numpy as np

def weight_variable_glorot1(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
#    tf.set_random_seed(123)
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32,seed = 12)
    return tf.Variable(initial, name=name)


def weight_variable_glorot2(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
#    tf.set_random_seed(231)
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32,seed = 50)
    return tf.Variable(initial, name=name)
