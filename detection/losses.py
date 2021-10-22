from tensorflow.keras import losses
import tensorflow as tf


def focal_loss(y_true, y_pred):
    gamma = 2.
    alpha = 0.25
    epsilon = 1e-7
    loss = y_true * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred + epsilon)
    loss = -alpha * tf.reduce_sum(loss, axis=-1)
    return tf.reduce_mean(loss)


def loc_losses(y_true, y_pred):
    l1_smooths = losses.huber(y_true, y_pred)
