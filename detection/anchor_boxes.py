from static_values.values import STEPS, object_names
from utils import box_utils
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


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
class PredictionDecoder:
    def __init__(self, anchor_boxes: AnchorBoxes):
        self.anchor_boxes = anchor_boxes
        self.boxes_variant = tf.convert_to_tensor([0.1, 0.1, 0.2, 0.2])

    def compute_bboxes(self, offsets, labels, score_threshold=0.2, iou_threshold=0.5):
        # offsets (batch, num_boxes, 4)
        # offsets *= self.boxes_variant
        anchor_boxes = tf.expand_dims(self.anchor_boxes.boxes, axis=0)
        cx = anchor_boxes[:, :, 0] + offsets[:, :, 0] * 0.1 * anchor_boxes[:, :, 2]
        cy = anchor_boxes[:, :, 1] + offsets[:, :, 1] * 0.1 * anchor_boxes[:, :, 3]
        w = tf.exp(offsets[:, :, 2] * 0.2) * anchor_boxes[:, :, 2]
        h = tf.exp(offsets[:, :, 3] * 0.2) * anchor_boxes[:, :, 3]
        bboxes = tf.stack([cx, cy, w, h], axis=-1)
        bboxes = tf.clip_by_value(bboxes, 1e-7, 1.)
        return self.get_best_bboxes(bboxes, labels, score_threshold, iou_threshold)

    def get_best_bboxes(self, bboxes, labels, score_threshold, iou_threshold):
        # bboxes: (batch, num_boxes, 4)
        # labels: (batch, num_boxes, num_classes + 1)
        labels_scores = tf.reduce_max(labels, axis=-1)
        labels_idx = tf.argmax(labels, axis=-1)
        labels_scores = tf.convert_to_tensor(labels_scores)
        results = []
        batch_size, _, _ = bboxes.get_shape()
        for i in range(batch_size):
            object_boxes_indices = tf.where(labels_idx[i, :] > 0)
            bboxes_pos = tf.gather_nd(bboxes[i, :, :], tf.expand_dims(object_boxes_indices, axis=1))
            bboxes_pos = tf.reshape(bboxes_pos, shape=(-1, 4))
            labels_scores_pos = tf.gather_nd(labels_scores[i, :], object_boxes_indices)
            labels_idx_pos = tf.gather_nd(labels_idx[i, :], object_boxes_indices)
            bboxes_corners = box_utils.center_to_corners(bboxes_pos)
            selected = tf.image.non_max_suppression(
                bboxes_corners,
                labels_scores_pos,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                max_output_size=1000
            )

            results.append({
                "bboxes": tf.gather(bboxes_pos, selected),
                "labels": tf.gather(labels_idx_pos, selected),
                "scores": tf.gather(labels_scores_pos, selected)
            })

        return results
