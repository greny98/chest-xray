import tensorflow as tf
from tensorflow.keras import layers, Sequential, Input, Model

from detection.feature_pyramid import FeaturePyramid, get_backbone
from static_values.values import object_names


def build_head(num_filters):
    head = Sequential([Input(shape=[None, None, 256])], name='name')
    for _ in range(4):
        head.add(layers.Conv2D(256, 3, padding="same"))
        head.add(layers.BatchNormalization(epsilon=1.001e-5))
        head.add(layers.ReLU())
    head.add(layers.Conv2D(num_filters, 3, padding="same"))
    return head


def ssd_head(features):
    num_classes = len(object_names)
    num_anchor_boxes = 9
    classify_head = build_head(num_anchor_boxes * num_classes)
    detect_head = build_head(num_anchor_boxes * 4)
    classes_out = []
    box_outputs = []
    for feature in features:
        box_outputs.append(layers.Reshape([-1, 4])(detect_head(feature)))
        classes_out.append(layers.Reshape([-1, num_classes])(classify_head(feature)))
    classes_out = layers.Concatenate(axis=1)(classes_out)
    box_outputs = layers.Concatenate(axis=1)(box_outputs)
    return [classes_out, box_outputs]


def create_ssd_model(backbone_weights=None):
    backbone = get_backbone(backbone_weights)
    pyramid = FeaturePyramid(backbone)
    outputs = ssd_head(pyramid.outputs)
    return Model(inputs=[pyramid.input], outputs=outputs)


def create_training_fn(model: Model):
    @tf.function
    def training_step(images, ):
        pass
