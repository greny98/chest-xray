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
        print(y_true, y_pred)
        # batch, num_boxes, num_classes
        gamma = 2.
        alpha = 0.25
        epsilon = 1e-7
        loss = -alpha * y_true * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred + epsilon)
        loss = tf.reduce_sum(loss, axis=-1)
        # Tính loss positive và negative
        positive_indices = tf.where(y_true[:, :, 0] == 0)
        negative_indices = tf.where(y_true[:, :, 0] == 1)
        batch_losses = []
        for i in range(batch_size):
            # positive loss
            batch_indices = tf.reshape(tf.where(positive_indices[:, 0] == i), shape=(-1,))
            batch_positive = tf.gather(positive_indices, batch_indices)
            loss_positive = tf.gather_nd(loss, batch_positive)
            # negative loss
            batch_indices = tf.reshape(tf.where(negative_indices[:, 0] == i), shape=(-1,))
            batch_negative = tf.gather(negative_indices, batch_indices)
            loss_negative = tf.gather_nd(loss, batch_negative)
            loss_negative = tf.sort(loss_negative, direction='DESCENDING')
            # lấy số mẫu negative gấp 3 lần positive
            n_neg = 3 * loss_positive.get_shape()[0]
            sample_loss = tf.reduce_sum(loss_positive) + tf.reduce_sum(loss_negative[:n_neg])
            batch_losses.append(sample_loss)
        return tf.reduce_mean(batch_losses), positive_indices

    return focal_loss


def create_l1_smooth_loss(batch_size=BATCH_SIZE):
    def l1_smooth_loss(y_true, y_pred, positive_indices):
        # Tính loss của positive boxes và top các negative có loss cao
        l1_smooths = []
        for i in range(batch_size):
            batch_indices = tf.reshape(tf.where(positive_indices[:, 0] == i), shape=(-1,1))
            batch_positive = tf.gather_nd(positive_indices, batch_indices)
            y_true_batch = tf.gather_nd(y_true, batch_positive)
            y_pred_batch = tf.gather_nd(y_pred, batch_positive)
            l1_smooths.append(losses.Huber(reduction=losses.Reduction.SUM)(y_true_batch, y_pred_batch))
        return tf.reduce_mean(l1_smooths)

    return l1_smooth_loss
