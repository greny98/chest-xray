import numpy as np
import tensorflow as tf

from static_values.values import l_diseases


def get_losses_weights(y):
    neg_weight = np.mean(y.astype(float), axis=0)
    pos_weight = 1 - neg_weight
    loss_weights = {}
    # print("\n===============================================================")
    # print("- Losses Weights (Pos, Neg):")
    for idx, label in enumerate(l_diseases):
        loss_weights[label] = (pos_weight[idx], neg_weight[idx])
        # print(f"\t+ {label}: {loss_weights[label]}")
    return loss_weights


def loss_weighted_fn(loss_weights, label):
    def loss_fn(y_true, y_pred):
        pos_weight = loss_weights[label][0]
        neg_weight = loss_weights[label][1]
        y_pred = tf.cast(y_pred, tf.float64)
        y_true = tf.cast(y_true, tf.float64)
        pos_loss = -pos_weight * y_true * tf.math.log(y_pred + 1e-7)
        neg_loss = -neg_weight * (1 - y_true) * tf.math.log(1 - y_pred + 1e-7)
        mean_loss = tf.reduce_mean(pos_loss + neg_loss)
        return tf.reduce_sum(mean_loss)

    return loss_fn


def create_losses(loss_weights):
    losses = []
    for label in l_diseases:
        losses.append(loss_weighted_fn(loss_weights, label))
    return losses
