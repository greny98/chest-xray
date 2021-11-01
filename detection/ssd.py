import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers as Opt
from detection.feature_pyramid import FeaturePyramid, get_backbone
from detection.losses import CreateLoss
from static_values.values import object_names


def build_head(feature, num_filters, name):
    for i in range(3):
        feature = layers.Conv2D(256, 3, padding="same", name=name + '_conv' + str(i))(feature)
        feature = layers.BatchNormalization(epsilon=1.001e-5)(feature)
        feature = layers.ReLU()(feature)
    feature = layers.Conv2D(num_filters, 3, padding="same", name=name + '_conv_out')(feature)
    return feature


def ssd_head(features):
    num_classes = len(object_names)
    num_anchor_boxes = 9
    classes_outs = []
    box_outputs = []
    for idx, feature in enumerate(features):
        classify_head = build_head(feature, num_anchor_boxes * num_classes, 'classify_head' + str(idx))
        detect_head = build_head(feature, num_anchor_boxes * 4, 'detect_head' + str(idx))
        box_outputs.append(layers.Reshape([-1, 4])(detect_head))
        classes_out = layers.Reshape([-1, num_classes])(classify_head)
        classes_outs.append(layers.Activation('sigmoid', name='classify_out' + str(idx))(classes_out))
    classes_outs = layers.Concatenate(axis=1)(classes_outs)
    box_outputs = layers.Concatenate(axis=1)(box_outputs)
    return layers.Concatenate()([box_outputs, classes_outs])


def create_ssd_model(backbone_weights=None):
    backbone = get_backbone(backbone_weights)
    pyramid = FeaturePyramid(backbone)
    outputs = ssd_head(pyramid.outputs)
    return Model(inputs=[pyramid.input], outputs=outputs)


def create_training_fn(model: Model, optimizer: Opt.Adam, decay=5e-5):
    compute_loss = CreateLoss()
    decay_layers = []
    for l in model.layers:
        if '_head' in l.name:
            decay_layers.append(l.name)

    def training_step(images, labels):
        with tf.GradientTape() as tape:
            pred_labels = model(images, training=True)
            cls_loss, loc_loss, total_losses = compute_loss(labels, pred_labels)

            kernel_variables = [model.get_layer(name).weights[0] for name in decay_layers]
            wd_penalty = decay * tf.reduce_sum([tf.reduce_sum(tf.square(k)) for k in kernel_variables])
            total_losses += wd_penalty
        grads = tape.gradient(total_losses, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return {"total_losses": total_losses,
                "classify_losses": cls_loss,
                "localize_losses": loc_loss}

    return training_step


def create_val_fn(model: Model):
    compute_loss = CreateLoss()

    def val_step(images, labels):
        pred_labels = model(images, training=False)
        cls_loss, loc_loss, total_losses = compute_loss(labels, pred_labels)
        return {"total_losses": total_losses,
                "classify_losses": cls_loss,
                "localize_losses": loc_loss}

    return val_step


def calc_loop(ds, step_fn, total_mean_fn, classify_mean_fn, localize_mean_fn, mode='training'):
    print("Processing....")
    for step, [X, labels] in enumerate(ds):
        losses = step_fn(X, labels)
        total_mean_fn(losses["total_losses"])
        classify_mean_fn(losses["classify_losses"])
        localize_mean_fn(losses["localize_losses"])

        if step % 1 == 0:
            print(f"\tLoss at step {step + 1}:", total_mean_fn.result().numpy())
            print(f"\t\t- Classification:", classify_mean_fn.result().numpy())
            print(f"\t\t- Localization:", localize_mean_fn.result().numpy())

    print(f"\tAverage {mode.capitalize()} Loss at step:", total_mean_fn.result().numpy())
    print(f"\t\t- Classification:", classify_mean_fn.result().numpy())
    print(f"\t\t- Localization:", localize_mean_fn.result().numpy())
    return total_mean_fn.result().numpy()
