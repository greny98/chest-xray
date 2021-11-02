from static_values.values import STEPS, object_names
from utils import box_utils
import tensorflow as tf
from tensorflow.keras import layers


class AnchorBoxes:
    def __init__(self, steps):
        self.steps = steps
        self.feature_widths = [1. / step for step in self.steps]
        self.aspect_ratios = [0.5, 1., 2.]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]
        self.num_anchors = len(self.aspect_ratios) * len(self.scales)
        self.boxes = self.gen_anchor_boxes()
        self.total_dims = self.get_total_dims()

    def get_total_dims(self):
        total_boxes = tf.convert_to_tensor(self.steps, dtype=tf.int32)
        total_boxes = tf.reduce_sum(tf.square(total_boxes))
        n_ar = len(self.aspect_ratios)
        n_scale = len(self.scales)
        return total_boxes * n_ar * n_scale * 4

    def gen_anchor_boxes(self):
        boxes = []
        for fw, n_step in zip(self.feature_widths, self.steps):
            for row in range(n_step):
                for col in range(n_step):
                    cx = (col + 0.5) * fw
                    cy = (row + 0.5) * fw
                    for scale in self.scales:
                        for ratio in self.aspect_ratios:
                            sqrt_ratio = tf.sqrt(ratio)
                            width = scale * sqrt_ratio * fw
                            height = scale * (1 / sqrt_ratio) * fw
                            boxes.append(tf.convert_to_tensor([cx, cy, width, height]))
        return tf.stack(boxes, axis=0)


# ======================================================================================================================
class LabelEncoder:
    def __init__(self, num_classes=len(object_names), steps=STEPS):
        self.anchor_boxes = AnchorBoxes(steps)
        self.num_classes = num_classes  # 0 for background
        self.variants = tf.convert_to_tensor([0.1, 0.1, 0.2, 0.2], tf.float32)

    def compute_offsets(self, matched_gt_boxes):
        center = (matched_gt_boxes[:, :2] - self.anchor_boxes.boxes[:, :2]) / self.anchor_boxes.boxes[:, 2:]
        size = tf.math.log(matched_gt_boxes[:, 2:] / matched_gt_boxes[:, 2:])
        return tf.concat([center, size], axis=-1) / self.variants

    def matching(self, gt_boxes, gt_classes, match_iou=0.5, ignore_iou=0.4):
        """
        Matching ground truth boxes and anchor boxes
        :param gt_boxes:
        :param gt_classes:
        :param match_iou:
        :param ignore_iou:
        :return: offsets, anchor_boxes_classes
        """
        gt_boxes = tf.convert_to_tensor(gt_boxes)
        iou_matrix = box_utils.calc_IoU(self.anchor_boxes.boxes, gt_boxes, mode='center')
        matched_gt_idx = tf.argmax(iou_matrix, axis=-1)
        best_iou = tf.reduce_max(iou_matrix, axis=-1)
        positive_mask = tf.greater_equal(best_iou, match_iou)
        negative_mask = tf.less(best_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))

        # Labeled for anchor boxes
        # Find anchor boxes that has iou >= iou_threshold
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        anchor_boxes_classes = tf.gather(gt_classes, matched_gt_idx)
        anchor_boxes_classes = tf.where(negative_mask, -1, anchor_boxes_classes)
        anchor_boxes_classes = tf.where(ignore_mask, -2, anchor_boxes_classes)
        # TODO: Compute offsets for anchor box from ground truth and return both classes and offsets
        offsets = self.compute_offsets(matched_gt_boxes)
        return tf.concat([offsets, tf.expand_dims(anchor_boxes_classes, axis=-1)], axis=-1)


# ======================================================================================================================
class PredictionDecoder(layers.Layer):
    def __init__(self, anchor_boxes: AnchorBoxes, **kwargs):
        super(PredictionDecoder, self).__init__(**kwargs)
        self.anchor_boxes = tf.expand_dims(anchor_boxes.boxes, axis=0)
        self._box_variance = tf.convert_to_tensor([0.1, 0.1, 0.2, 0.2])
        self.confidence_threshold = 0.05,
        self.nms_iou_threshold = 0.5,
        self.max_detections_per_class = 100,
        self.max_detections = 100,

    def _decode_box_predictions(self, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * self.anchor_boxes[:, :, 2:] + self.anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * self.anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = box_utils.center_to_corners(boxes)
        return boxes_transformed

    def call(self, predictions, **kwargs):
        box_predictions = predictions[:, :, :4]
        cls_predictions = predictions[:, :, 4:]
        boxes = self._decode_box_predictions(box_predictions)
        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            100,
            100,
            0.5,
            0.03,
            # clip_boxes=False,
        )
