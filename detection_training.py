from keras.models import Model

from backbone.model import load_basenet
from detection.anchor_boxes import AnchorBoxes, LabelEncoder, PredictionDecoder
import tensorflow as tf
from tensorflow.keras.utils import plot_model

from detection.feature_pyramid import FeaturePyramid, get_backbone
from detection.losses import create_focal_loss, create_loc_loss

from tensorflow.keras.utils import to_categorical

from detection.ssd import ssd_head, create_ssd_model

# pyramid = FeaturePyramid(get_backbone())
# outputs = ssd_head(pyramid.outputs)
# model = Model(inputs=[pyramid.input], outputs=outputs)
# model.summary()
# pyramid = FeaturePyramid(backbone)
# from detection.losses import focal_loss
#
# label_encoder = LabelEncoder(num_classes=2)
# gt_boxes = tf.convert_to_tensor([
#     [0.2, 0.2, 0.5, 0.5],
#     [0.06, 0.06, 0.03, 0.05],
#     [0.3, 0.3, 0.15, 0.15]])
# gt_classes = tf.convert_to_tensor([1, 2, 1])
#
# offsets, classes = label_encoder.matching(gt_boxes, gt_classes)
# print(offsets)

# rand = tf.random.uniform(shape=(2, 3, 3), dtype=tf.float64)
# c_pred = tf.nn.softmax(rand, axis=-1)
# c_true = tf.convert_to_tensor([
#     [
#         [0, 0, 1],
#         [1, 0, 0],
#         [1, 0, 0],
#     ],
#     [
#         [0, 1, 0],
#         [1, 0, 0],
#         [0, 1, 0],
#     ]
#
# ], tf.float64)
#
# loc_pred = tf.convert_to_tensor([
#     [
#         [0, 0, 0.5, 0.5],
#         [0.1, 0.1, 0.2, 0.3],
#         [0.25, 0.1, 0.35, 0.25],
#     ],
#     [
#         [0.15, 0.1, 0.3, 0.3],
#         [0.1, 0.3, 0.2, 0.3],
#         [0.25, 0.25, 0.15, 0.25],
#     ]
# ], tf.float64)
# loc_true = tf.convert_to_tensor([
#     [
#         [0.1, 0.1, 0.4, 0.4],
#         [0.15, 0.21, 0.21, 0.33],
#         [0.2, 0.12, 0.5, 0.35],
#     ],
#     [
#         [0.15, 0.1, 0.3, 0.3],
#         [0.1, 0.1, 0.4, 0.15],
#         [0.35, 0.15, 0.25, 0.25],
#     ]
# ], tf.float64)
#
# focal_loss_fn = create_focal_loss(batch_size=2)
# loc_loss_fn = create_loc_loss(batch_size=2)
# conf_loss, pos_indices = focal_loss_fn(c_true, c_pred)
# loc_loss = loc_loss_fn(loc_true, loc_pred, pos_indices)
# print(conf_loss, loc_loss)

model = create_ssd_model()
model.summary()
