from detection.anchor_boxes import AnchorBoxes, LabelEncoder
import tensorflow as tf

from tensorflow.keras.utils import to_categorical

# backbone = get_backbone(weights='ckpt/checkpoint')
# pyramid = FeaturePyramid(backbone)
# label_encoder = LabelEncoder(n_classes=3)
# gt_boxes = tf.convert_to_tensor([[0.2, 0.2, 0.5, 0.5], [0.06, 0.06, 0.03, 0.05]])
# gt_classes = tf.convert_to_tensor([1, 2])
# label_encoder.matching(gt_boxes, gt_classes)
