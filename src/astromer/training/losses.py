import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.nn import (sigmoid_cross_entropy_with_logits,
                           softmax_cross_entropy_with_logits)

def custom_rmse(y_true, y_pred, sample_weight=None, mask=None):
    inp_shp = tf.shape(y_true)
    residuals = tf.square(y_true - y_pred)

    if sample_weight is not None:
        residuals = tf.multiply(residuals, sample_weight)

    if mask is not None:
        residuals = tf.multiply(residuals, mask)

    residuals  = tf.reduce_sum(residuals, 1)
    mse_mean = tf.math.divide_no_nan(residuals,
                                     tf.reduce_sum(mask, 1))

    mse_mean = tf.reduce_mean(mse_mean)
    return tf.math.sqrt(mse_mean)
