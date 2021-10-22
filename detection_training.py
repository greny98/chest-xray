from detection.anchor_boxes import AnchorBoxes, LabelEncoder
import tensorflow as tf

from tensorflow.keras.utils import to_categorical

# backbone = get_backbone(weights='ckpt/checkpoint')
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
from detection.losses import create_focal_loss

rand = tf.random.uniform(shape=(2, 3, 3), dtype=tf.float64)
y_pred = tf.nn.softmax(rand, axis=-1)
y_true = tf.convert_to_tensor([
    [
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
    ],
    [
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
    ]

], tf.float64)
focal_loss = create_focal_loss(batch_size=2)
conf_loss = focal_loss(y_true, rand)
