from tensorflow.keras import Model, layers, regularizers, losses
from tensorflow.keras.applications import densenet
import tensorflow as tf
from siim.configs import IMAGE_CLS_SIZE, LABELS
import math


class SPPLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(SPPLayer, self).__init__(**kwargs)

    @staticmethod
    def _spp_pool(inputs, ksize):
        _, h, w, c = inputs.shape
        win_h = math.ceil(h / ksize)
        win_w = math.ceil(w / ksize)
        stride_h = math.floor(h / ksize)
        stride_w = math.floor(w / ksize)
        x = tf.nn.max_pool2d(inputs,
                             padding='VALID',
                             ksize=(win_h, win_w),
                             strides=(stride_h, stride_w))
        return tf.reshape(x, shape=(-1, ksize * ksize * c))

    def call(self, inputs, *args, **kwargs):
        _, h, w, c = inputs.shape
        # 4x4, 3x3, 2x2, 1x1
        lv1 = self._spp_pool(inputs, ksize=4)
        lv2 = self._spp_pool(inputs, ksize=3)
        lv3 = self._spp_pool(inputs, ksize=2)
        lv4 = self._spp_pool(inputs, ksize=1)
        return tf.concat([lv1, lv2, lv3, lv4], axis=-1)


class DiagnosisModel(Model):
    def __init__(self, basenet_ckpt=None, l2_decay=2e-5, cnn_trainable=True, **kwargs):
        super(DiagnosisModel, self).__init__(name='DiagnosisModel', **kwargs)
        IMAGE_SIZE = IMAGE_CLS_SIZE
        self.basenet = densenet.DenseNet201(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                            include_top=False, weights='imagenet')
        if basenet_ckpt is not None:
            self.basenet.load_weights(basenet_ckpt).expect_partial()
        self.basenet.trainable = cnn_trainable
        # More layers
        # self.global_avg_pools = layers.GlobalAveragePooling2D(name='global_avg_pool')
        self.conv256 = layers.Conv2D(256, 1, name='conv256')
        self.spp = SPPLayer()
        self.dropout = layers.Dropout(0.3)
        self.dense = layers.Dense(512, name='dense', kernel_regularizer=regularizers.l2(l2_decay))
        self.bn = layers.BatchNormalization(epsilon=1.001e-5)
        self.relu = layers.ReLU()
        self.dense_out = layers.Dense(len(LABELS), name='dense_out',
                                      kernel_regularizer=regularizers.l2(l2_decay))

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        x = self.basenet(inputs)
        x = self.conv256(x)
        x = self.spp(x)
        x = self.dense(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.dropout(x, training=training)
        x = self.dense_out(x)
        return x


class FocalLoss(losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.):
        super(FocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        loss = tf.reduce_sum(loss, axis=-1)
        return tf.reduce_mean(loss)
