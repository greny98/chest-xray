from static_values.values import STEPS, BATCH_SIZE, object_names
from utils import box_utils
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


class AnchorBoxes:
    def __init__(self, steps):
        self.steps = steps
        self.feature_widths = [1. / step for step in self.steps]
        self.aspect_ratios = [0.5, 1., 2.]
        self.scales = [2 ** x for x in [0, 1 / 4, 2 / 4, 3 / 4]]
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
        total = 0
        for f_num, n_step in enumerate(self.steps):
            for row in range(n_step):
                for col in range(n_step):
                    cx = (col + 0.5) * self.feature_widths[f_num]
                    cy = (row + 0.5) * self.feature_widths[f_num]
                    for scale in self.scales:
                        for ratio in self.aspect_ratios:
                            sqrt_ratio = tf.sqrt(ratio)
                            width = scale * sqrt_ratio * self.feature_widths[f_num]
                            height = scale * (1 / sqrt_ratio) * self.feature_widths[f_num]
                            boxes.append(tf.convert_to_tensor([cx, cy, width, height]))
                            total += 1
        return tf.stack(boxes, axis=0)


# ======================================================================================================================
class LabelEncoder:
    def __init__(self, num_classes=len(object_names), steps=STEPS):
        self.anchor_boxes = AnchorBoxes(steps)
        self.num_classes = num_classes + 1  # 0 for background

    def compute_offsets(self, matched_gt_boxes):
        off_cx = (matched_gt_boxes[:, 0] - self.anchor_boxes.boxes[:, 0]) / self.anchor_boxes.boxes[:, 2]
        off_cy = (matched_gt_boxes[:, 1] - self.anchor_boxes.boxes[:, 1]) / self.anchor_boxes.boxes[:, 3]
        off_w = tf.math.log(matched_gt_boxes[:, 2] / self.anchor_boxes.boxes[:, 2])
        off_h = tf.math.log(matched_gt_boxes[:, 3] / self.anchor_boxes.boxes[:, 3])
        return tf.stack([off_cx, off_cy, off_w, off_h], axis=1)

    def matching(self, gt_boxes, gt_classes, iou_threshold=0.4):
        """
        Matching ground truth boxes and anchor boxes
        :param gt_boxes:
        :param gt_classes:
        :param iou_threshold:
        :return:
        """
        iou_anchor2gt = []
        for idx, gt_box in enumerate(gt_boxes):
            # Tính IoU của anchor_boxes với tất cả gt_boxes
            iou = box_utils.calc_IoU(tf.expand_dims(gt_box, axis=0),
                                     self.anchor_boxes.boxes, mode='center')
            iou_anchor2gt.append(iou)
        iou_anchor2gt = tf.stack(iou_anchor2gt, axis=1)
        # Get best IoU
        rows = tf.range(0, self.anchor_boxes.total_dims / 4, dtype=tf.int32)
        arg_max_iou = tf.argmax(iou_anchor2gt, axis=1)
        arg_max_iou = tf.cast(arg_max_iou, tf.int32)
        best_iou = tf.gather_nd(
            iou_anchor2gt,
            tf.stack([rows, arg_max_iou], axis=1))
        # Labeled for anchor boxes
        # Find anchor boxes that has iou >= iou_threshold
        negative_indices = tf.cast(tf.where(best_iou < iou_threshold), tf.int32)
        anchor_boxes_classes = tf.tensor_scatter_nd_update(
            tf.expand_dims(tf.gather(gt_classes, arg_max_iou), axis=1),
            negative_indices,
            tf.zeros_like(negative_indices))
        anchor_boxes_classes = tf.reshape(anchor_boxes_classes, shape=(-1,))
        matched_gt_boxes = tf.gather_nd(gt_boxes, tf.expand_dims(arg_max_iou, axis=-1))
        # TODO: Compute offsets for anchor box from ground truth and return both classes and offsets
        offsets = self.compute_offsets(matched_gt_boxes)
        return offsets, to_categorical(anchor_boxes_classes, self.num_classes)


# ======================================================================================================================
class PredictionDecoder:
    def __init__(self, anchor_boxes: AnchorBoxes):
        self.anchor_boxes = anchor_boxes

    def compute_bboxes(self, offsets, labels, score_threshold=0.6, iou_threshold=0.5):
        # offsets (batch, num_boxes, 4)
        anchor_boxes = tf.expand_dims(self.anchor_boxes.boxes, axis=0)
        cx = anchor_boxes[:, :, 0] + offsets[:, :, 0] * anchor_boxes[:, :, 2]
        cy = anchor_boxes[:, :, 1] + offsets[:, :, 1] * anchor_boxes[:, :, 3]
        w = tf.exp(offsets[:, :, 2]) * anchor_boxes[:, :, 2]
        h = tf.exp(offsets[:, :, 3]) * anchor_boxes[:, :, 3]
        bboxes = tf.stack([cx, cy, w, h], axis=-1)
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

            selected = tf.image.non_max_suppression(
                bboxes_pos,
                labels_scores_pos,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                max_output_size=5
            )
            results.append({
                "bboxes": tf.gather(bboxes_pos, selected),
                "labels": tf.gather(labels_idx_pos, selected)
            })
        return results
