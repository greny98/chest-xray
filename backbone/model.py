from tensorflow.keras import layers, Model
from tensorflow.keras.applications import densenet
import tensorflow as tf

from static_values.values import IMAGE_SIZE, l_diseases


def load_basenet(input_shape):
    return densenet.DenseNet169(input_shape=input_shape, include_top=False, weights='imagenet')


def build_top(base_net: Model):
    features = layers.GlobalAveragePooling2D()(base_net.output)
    full_outputs = []
    for disease in l_diseases:
        full_outputs.append(
            layers.Dense(1, activation='sigmoid', name=disease)(features))
    return full_outputs


def create_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)):
    base_net = load_basenet(input_shape=input_shape)
    outputs = build_top(base_net)
    return Model(inputs=[base_net.inputs], outputs=outputs)


def create_training_step(model: Model, l_losses, l_metrics, optimizer, decay=5.0e-5):
    @tf.function
    def training_step(X, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(X, training=True)
            total_losses = 0.
            for idx, (each_y_pred, each_y_true) in enumerate(zip(y_pred, y_true)):
                each_y_pred = tf.reshape(each_y_pred, shape=(-1,))
                current_loss = l_losses[idx](each_y_true, each_y_pred)
                total_losses += current_loss
                l_metrics[idx](each_y_true, each_y_pred)
            # Calculate weight decay
            kernel_variables = [model.get_layer(name).weights[0] for name in l_diseases]
            wd_penalty = decay * tf.reduce_sum([tf.reduce_sum(tf.square(k)) for k in kernel_variables])
            wd_penalty = tf.cast(wd_penalty, tf.float64)
            total_losses += wd_penalty
        grads = tape.gradient(total_losses, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return total_losses

    return training_step


def create_validate_step(model: Model, l_losses, l_metrics):
    @tf.function
    def validate_step(X, y_true):
        y_pred = model(X, training=False)
        total_losses = 0.
        for idx, (each_y_pred, each_y_true) in enumerate(zip(y_pred, y_true)):
            each_y_pred = tf.reshape(each_y_pred, shape=(-1,))
            total_losses = total_losses + l_losses[idx](each_y_true, each_y_pred)
            l_metrics[idx](each_y_true, each_y_pred)
        return total_losses

    return validate_step


def calc_loop(ds, step_fn, mean_loss_fn, metrics_fn, mode='training'):
    print("Processing....")
    for step, (X, y) in enumerate(ds):
        losses = step_fn(X, y)
        mean_loss_fn(losses)
        if step % 1000 == 0:
            print(f"\t- Loss at step {step + 1}:", mean_loss_fn.result().numpy())
    print(f"\t- {mode.capitalize()} Loss: ", mean_loss_fn.result().numpy())
    print(f"\t- {mode.capitalize()} Accuracy: ")
    avg_acc = 0.
    for i in range(len(metrics_fn)):
        acc = metrics_fn[i].result().numpy()
        avg_acc += acc
        print(f"\t\t+ {l_diseases[i]}: ", acc)
    print(f"\t\t+ Average {mode.capitalize()} Accuracy: ", avg_acc / len(metrics_fn))
    return mean_loss_fn.result().numpy(), avg_acc
