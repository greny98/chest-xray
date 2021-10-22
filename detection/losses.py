from tensorflow.keras import losses
import tensorflow as tf


def create_focal_loss(batch_size):
    """
    Create focal loss
        - TODO: Tính focal loss theo công thức cho toàn bộ batch
        - TODO: Chọn
    :param batch_size:
    :return:
    """
    def focal_loss(y_true, y_pred):
        # batch, num_boxes, num_classes
        gamma = 2.
        alpha = 0.25
        epsilon = 1e-7
        loss = -alpha * y_true * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred + epsilon)
        loss = tf.reduce_sum(loss, axis=-1)
        print("==== loss", loss)
        # Tìm loss positive
        positive_indices = tf.where(y_true[:, :, 0] == 0)
        total_losses = 0.
        for i in range(batch_size):
            batch_indices = tf.reshape(tf.where(positive_indices[:, 0] == i), shape=(-1,))
            batch_positives = tf.gather(positive_indices, batch_indices)
            loss_positive = tf.gather_nd(loss, batch_positives)
            total_losses += tf.reduce_sum(loss_positive)
        # Tìm loss negative

        # Chọn loss negative cao nhất

        return tf.reduce_mean(loss)

    return focal_loss


def loc_losses(y_true, y_pred):
    l1_smooths = losses.huber(y_true, y_pred)
