import tensorflow as tf


def corners_to_center(corners):
    """
    Convert boxes from (xmin,ymin,xmax,ymax) to (x,y,w,h)
    :param corners:
    :return:
    """
    x = (corners[:, 0] + corners[:, 2]) / 2
    y = (corners[:, 1] + corners[:, 3]) / 2
    w = corners[:, 2] - corners[:, 0]
    h = corners[:, 3] - corners[:, 1]
    return tf.stack([x, y, w, h], axis=1)


def center_to_corners(center):
    """
    Convert boxes from (x,y,w,h) to (xmin,ymin,xmax,ymax)
    :param center:
    :return:
    """
    xmin = center[:, 0] - center[:, 2] / 2
    ymin = center[:, 1] - center[:, 3] / 2
    xmax = center[:, 0] + center[:, 2] / 2
    ymax = center[:, 1] + center[:, 3] / 2
    return tf.stack([xmin, ymin, xmax, ymax], axis=1)


def calc_IoU(boxes1, boxes2, mode='corner', reduce_mean=False):
    """
    Compute IoU of predictions and ground_truth
        (Use for corners)
    :param boxes1:
    :param boxes2:
    :param mode:
    :param reduce_mean:
    :return:
    """
    if mode == 'center':
        boxes1 = center_to_corners(boxes1)
        boxes2 = center_to_corners(boxes2)

    # Calculate Intersection
    inter_xmin, inter_ymin = tf.maximum(boxes1[:, 0], boxes2[:, 0]), tf.maximum(boxes1[:, 1], boxes2[:, 1])
    inter_xmax, inter_ymax = tf.minimum(boxes1[:, 2], boxes2[:, 2]), tf.minimum(boxes1[:, 3], boxes2[:, 3])
    inter_w = tf.maximum(0., inter_xmax - inter_xmin)
    inter_h = tf.maximum(0., inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    # Calculate Union
    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = tf.maximum(boxes1_area + boxes2_area - inter_area, 1e-7)

    # IoU
    IoU = tf.clip_by_value(inter_area / union_area, 0., 1.)
    if reduce_mean:
        return tf.reduce_mean(IoU)
    return IoU
