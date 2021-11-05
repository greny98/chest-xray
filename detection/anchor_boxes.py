from static_values.values import object_names
from utils import box_utils
import tensorflow as tf
from tensorflow.keras import layers


class AnchorBoxes:
    def __init__(self):
        self._aspect_ratios = [0.5, 1.0, 2.0]
        self._scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]
        self._num_anchors = len(self._aspect_ratios) * len(self._scales)
        self._areas = [x ** 2 for x in [16, 32, 64, 128, 256, 512]]
        self._strides = [2 ** i for i in range(3, 8)]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self._aspect_ratios:
                # Tính width và height ứng với mỗi area
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1),
                    shape=[1, 1, 2]
                )
                for scale in self._scales:
                    # Tính cho mỗi scale
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_width, feature_height, level):
        # level bắt đầu từ 3
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(self._anchor_dims[level - 3], [feature_width, feature_height, 1, 1])
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(anchors, shape=(-1, 4))

    def create_anchors_boxes(self, image_width, image_height):
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_width / 2 ** i),
                tf.math.ceil(image_height / 2 ** i),
                i)
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)


# ======================================================================================================================
class LabelEncoder:
    def __init__(self, num_classes=len(object_names)):
        self._anchor_boxes = AnchorBoxes()
        self._num_classes = num_classes
        self._box_variance = tf.convert_to_tensor([0.1, 0.1, 0.2, 0.2], tf.float32)

    @staticmethod
    def _match_anchor_boxes(anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4):
        iou_matrix = box_utils.calc_IoU(anchor_boxes, gt_boxes, mode='center')
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:])
            ],
            axis=-1
        )
        box_target = box_target / self._box_variance
        return box_target

    def encode_sample(self, image_shape, gt_boxes, cls_ids):
        anchor_boxes = self._anchor_boxes.create_anchors_boxes(image_shape[1], image_shape[0])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(anchor_boxes, gt_boxes)
        # Compute boxes
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        # Compute cls
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(tf.equal(positive_mask, 1.0), matched_gt_cls_ids, -1.0)
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([box_target, cls_target], axis=-1)
        return label


# ======================================================================================================================
class PredictionDecoder(layers.Layer):
    def __init__(self, **kwargs):
        super(PredictionDecoder, self).__init__(**kwargs)
        self._box_variance = tf.convert_to_tensor([0.1, 0.1, 0.2, 0.2])
        self._anchor_boxes = AnchorBoxes()

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = box_utils.center_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_boxes.create_anchors_boxes(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)
        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            max_output_size_per_class=10,
            max_total_size=10,
            iou_threshold=0.5,
            score_threshold=0.05,
            clip_boxes=False,
        )
