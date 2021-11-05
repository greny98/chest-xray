from tensorflow.keras import Model, layers, regularizers, losses
from tensorflow.keras.applications import densenet
import tensorflow as tf
from siim.configs import IMAGE_CLS_SIZE, LABELS


class DiagnosisModel(Model):
    def __init__(self, basenet_ckpt=None, l2_decay=2e-5, cnn_trainable=True, **kwargs):
        super(DiagnosisModel, self).__init__(kwargs, name='DiagnosisModel')
        IMAGE_SIZE = IMAGE_CLS_SIZE
        self.basenet = densenet.DenseNet201(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                            include_top=False, weights='imagenet')
        if basenet_ckpt is not None:
            self.basenet.load_weights(basenet_ckpt).expect_partial()
        self.basenet.trainable = cnn_trainable
        # More layers
        self.global_avg_pools = layers.GlobalAveragePooling2D(name='global_avg_pool')
        self.dropout = layers.Dropout(0.3)
        self.dense_out = layers.Dense(len(LABELS), name='dense_out',
                                      kernel_regularizer=regularizers.l2(l2_decay))

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        x = self.basenet(inputs)
        x = self.global_avg_pools(x)
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
