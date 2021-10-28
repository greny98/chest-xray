import tensorflow as tf
from tensorflow.keras import layers, Sequential, Input, Model, optimizers as Opt
from detection.feature_pyramid import FeaturePyramid, get_backbone
from detection.losses import create_focal_loss, create_l1_smooth_loss
from static_values.values import object_names


def build_head(num_filters):
    head = Sequential([Input(shape=[None, None, 256])])
    for _ in range(4):
        head.add(layers.Conv2D(256, 3, padding="same"))
        head.add(layers.BatchNormalization(epsilon=1.001e-5))
        head.add(layers.ReLU())
    head.add(layers.Conv2D(num_filters, 3, padding="same"))
    return head


def ssd_head(features):
    num_classes = len(object_names) + 1
    num_anchor_boxes = 12
    classify_head = build_head(num_anchor_boxes * num_classes)
    detect_head = build_head(num_anchor_boxes * 4)
    classes_outs = []
    box_outputs = []
    for feature in features:
        box_outputs.append(layers.Reshape([-1, 4])(detect_head(feature)))
        classes_out = layers.Reshape([-1, num_classes])(classify_head(feature))
        classes_outs.append(layers.Softmax(axis=-1)(classes_out))
    classes_outs = layers.Concatenate(axis=1)(classes_outs)
    box_outputs = layers.Concatenate(axis=1)(box_outputs)
    return [classes_outs, box_outputs]


def create_ssd_model(backbone_weights=None):
    backbone = get_backbone(backbone_weights)
    pyramid = FeaturePyramid(backbone)
    outputs = ssd_head(pyramid.outputs)
    return Model(inputs=[pyramid.input], outputs=outputs)


def create_training_fn(model: Model, optimizer: Opt.Adam):
    focal_loss = create_focal_loss()
    l1_smooth_loss = create_l1_smooth_loss()

    def training_step(images, offsets, labels_oh):
        with tf.GradientTape() as tape:
            pred_labels, pred_offsets = model(images, training=True)
            classify_losses, pos_indices = focal_loss(labels_oh, pred_labels)
            localize_losses = l1_smooth_loss(offsets, pred_offsets, pos_indices)
            total_losses = classify_losses + localize_losses
        grads = tape.gradient(total_losses, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return {"total_losses": total_losses,
                "classify_losses": classify_losses,
                "localize_losses": localize_losses}

    return training_step


def create_val_fn(model: Model):
    focal_loss = create_focal_loss()
    l1_smooth_loss = create_l1_smooth_loss()

    def val_step(images, offsets, labels_oh):
        pred_labels, pred_offsets = model(images, training=False)
        classify_losses, pos_indices = focal_loss(labels_oh, pred_labels)
        localize_losses = l1_smooth_loss(offsets, pred_offsets, pos_indices)
        total_losses = classify_losses + localize_losses
        return {"total_losses": total_losses,
                "classify_losses": classify_losses,
                "localize_losses": localize_losses}

    return val_step


def calc_loop(ds, step_fn, total_mean_fn, classify_mean_fn, localize_mean_fn, mode='training'):
    print("Processing....")
    for step, [X, offsets, labels_oh] in enumerate(ds):
        losses = step_fn(X, offsets, labels_oh)
        total_mean_fn(losses["total_losses"])
        classify_mean_fn(losses["classify_losses"])
        localize_mean_fn(losses["localize_losses"])

        if step % 100 == 0:
            print(f"\tLoss at step {step + 1}:", total_mean_fn.result().numpy())
            print(f"\t\t- Classification:", classify_mean_fn.result().numpy())
            print(f"\t\t- Localization:", localize_mean_fn.result().numpy())

    print(f"\tAverage {mode.capitalize()} Loss at step:", total_mean_fn.result().numpy())
    print(f"\t\t- Classification:", classify_mean_fn.result().numpy())
    print(f"\t\t- Localization:", localize_mean_fn.result().numpy())
    return total_mean_fn.result().numpy()
