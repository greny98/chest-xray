from detection.anchor_boxes import AnchorBoxes, LabelEncoder
import tensorflow as tf

from tensorflow.keras.utils import to_categorical

# backbone = get_backbone(weights='ckpt/checkpoint')
# pyramid = FeaturePyramid(backbone)
from detection.losses import focal_loss

label_encoder = LabelEncoder(num_classes=2)
gt_boxes = tf.convert_to_tensor([
    [0.2, 0.2, 0.5, 0.5],
    [0.06, 0.06, 0.03, 0.05],
    [0.3, 0.3, 0.15, 0.15]])
gt_classes = tf.convert_to_tensor([1, 2, 1])

offsets, classes = label_encoder.matching(gt_boxes, gt_classes)
rand = tf.random.uniform(shape=(label_encoder.anchor_boxes.total_dims // 4, 3))
rand = tf.nn.softmax(rand)
conf_loss = focal_loss(classes, rand)
print(classes[0], rand[0])
print(conf_loss[0])
# tensor = tf.convert_to_tensor([
#     [1, 3, 5],
#     [3, 4, 2],
#     [2, 3, 4],
# ], tf.float64)
#
# gt_classes = tf.convert_to_tensor([1, 2])
# print(tf.gather(gt_classes, [0, 0, 1]))
