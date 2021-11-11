from tensorflow.keras import layers, Model, regularizers
from detection.feature_pyramid import FeaturePyramid, get_backbone
from static_values.values import object_names

l2 = regularizers.l2(2.5e-5)


def build_head(feature, num_filters, name):
    for i in range(4):
        feature = layers.Conv2D(256, 3, padding="same", name=name + '_conv' + str(i))(feature)
        feature = layers.BatchNormalization(epsilon=1.001e-5)(feature)
        feature = layers.ReLU()(feature)
    feature = layers.Conv2D(num_filters, 3, padding="same", name=name + '_conv_out', kernel_regularizer=l2)(feature)
    return feature


def ssd_head(features):
    # num_classes = len(object_names)
    num_classes = 1
    num_anchor_boxes = 9
    classes_outs = []
    box_outputs = []
    for idx, feature in enumerate(features):
        classify_head = build_head(feature, num_anchor_boxes * num_classes, 'classify_head' + str(idx))
        detect_head = build_head(feature, num_anchor_boxes * 4, 'detect_head' + str(idx))
        box_outputs.append(layers.Reshape([-1, 4])(detect_head))
        classes_out = layers.Reshape([-1, num_classes])(classify_head)
        classes_outs.append(classes_out)
    classes_outs = layers.Concatenate(axis=1)(classes_outs)
    box_outputs = layers.Concatenate(axis=1)(box_outputs)
    return layers.Concatenate(axis=-1)([box_outputs, classes_outs])


def create_ssd_model(backbone_weights=None):
    backbone = get_backbone(backbone_weights)
    pyramid = FeaturePyramid(backbone)
    outputs = ssd_head(pyramid.outputs)
    return Model(inputs=[pyramid.input], outputs=outputs)
