from tensorflow.keras import losses
import tensorflow as tf

from static_values.values import BATCH_SIZE


def create_focal_loss(batch_size=BATCH_SIZE):
    """
    Create focal loss
        - Tính focal loss theo công thức cho toàn bộ batch
        - Chọn positive loss, negative loss cho mỗi ảnh (1:3)
        - Tính tổng loss cho mỗi ảnh
        - Tính mean của batch
    :param batch_size:
    :return: focal_loss
    """

    def focal_loss(y_true, y_pred):
        # print(y_true, y_pred)
        # batch, num_boxes, num_classes
        tf.debugging.check_numerics(y_pred, message="INF")
        gamma = 2.
        alpha = 0.25
        epsilon = 1e-8
        # Calculate
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma) * y_true
        loss = alpha * weight * ce
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.reduce_sum(loss, axis=-1)
        # Tính loss positive và negative
        positive_indices = tf.where(y_true[:, :, 0] == 0)
        return tf.reduce_mean(loss), positive_indices

    return focal_loss


def create_l1_smooth_loss(batch_size):
    def l1_smooth_loss(y_true, y_pred, positive_indices):
        # Tính loss của positive boxes và top các negative có loss cao
        l1_smooths = []
        for i in range(batch_size):
            batch_indices = tf.reshape(tf.where(positive_indices[:, 0] == i), shape=(-1, 1))
            batch_positive = tf.gather_nd(positive_indices, batch_indices)
            y_true_batch = tf.gather_nd(y_true, batch_positive)
            y_pred_batch = tf.gather_nd(y_pred, batch_positive)
            l1_smooths.append(losses.Huber(reduction=losses.Reduction.SUM)(y_true_batch, y_pred_batch))
        return tf.reduce_mean(l1_smooths)

    return l1_smooth_loss
