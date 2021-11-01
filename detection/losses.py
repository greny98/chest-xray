import tensorflow as tf

from static_values.values import object_names


def CreateClassificationLoss(_gamma=2., _alpha=0.25):
    def focal_loss(y_true, y_pred):
        # Calculate
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        alpha = tf.where(tf.equal(y_true, 1.0), _alpha, (1.0 - _alpha))
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1 - y_pred)
        loss = alpha * tf.pow(1.0 - pt, _gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    return focal_loss


def CreateLocalizationLoss(delta=1):
    def l1_smooth_loss(y_true, y_pred):
        # Tính loss của positive boxes và top các negative có loss cao
        diff = y_true - y_pred
        abs_diff = tf.abs(diff)
        square_diff = tf.square(diff)
        loss = tf.where(tf.less(abs_diff, delta), 0.5 * square_diff, abs_diff - 0.5)
        return tf.reduce_sum(loss, axis=-1)

    return l1_smooth_loss


def CreateLoss(num_classes=len(object_names)):
    def compute_loss(y_true, y_pred):
        focal_loss = CreateClassificationLoss()
        l1_smooth_loss = CreateLocalizationLoss()
        y_pred = tf.cast(y_pred, tf.float32)
        # Create mask
        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1), tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2), tf.float32)
        # Get classification loss
        cls_true = tf.one_hot(
            tf.cast(y_true[:, :, 4], tf.int32),
            depth=num_classes,
            dtype=tf.float32,
        )
        cls_pred = y_pred[:, :, 4:]
        cls_loss = focal_loss(cls_true, cls_pred)
        cls_loss = tf.where(tf.equal(ignore_mask, 1.), 0, cls_loss)
        # bboxes losses
        loc_loss = l1_smooth_loss(y_true[:, :, :4], y_pred[:, :, :4])
        loc_loss = tf.where(tf.equal(positive_mask, 1.), loc_loss, 0.)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        cls_loss = tf.math.divide_no_nan(tf.reduce_sum(cls_loss, axis=-1), normalizer)
        loc_loss = tf.math.divide_no_nan(tf.reduce_sum(loc_loss, axis=-1), normalizer)
        return cls_loss, loc_loss, cls_loss + loc_loss

    return compute_loss
