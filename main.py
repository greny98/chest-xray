import os
import argparse
from tensorflow.keras import optimizers, metrics
from backbone.losses import get_losses_weights, create_losses
from backbone.model import create_model, create_training_step, create_validate_step, calc_loop
from utils.data_generator import create_ds
from utils.dataframe import read_csv, train_val_split


def reset_states():
    global train_mean_losses
    global val_mean_losses
    global training_metrics
    global val_metrics
    train_mean_losses.reset_states()
    val_mean_losses.reset_states()
    for i in range(len(training_metrics)):
        training_metrics[i].reset_states()
    for i in range(len(val_metrics)):
        val_metrics[i].reset_states()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--csv_file', type=str)
    parser.add_argument('--image_dir', type=str, default='images')
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--output_dir', type=str, default='model')
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if not os.path.exists(args['output_dir']):
        os.mkdir(args['output_dir'])
    images_dir = args['image_dir']
    # Load csv
    X_train_val_df, y_train_val_df = read_csv(args['csv_file'])
    # Split train, val
    (X_train, y_train), (X_val, y_val) = train_val_split(X_train_val_df.values, y_train_val_df.values, log=False)
    # Flatten X
    X_train = X_train.reshape(-1)
    X_val = X_val.reshape(-1)
    # Create ds
    train_ds = create_ds(X_train, y_train, images_dir, training=True)
    val_ds = create_ds(X_val, y_val, images_dir)
    # Create model
    model = create_model()
    # Compile with loss
    losses_weights = get_losses_weights(y_train)
    l_losses = create_losses(losses_weights)
    # training
    EPOCHS = args['epochs']
    lr = args['lr']
    lr_decay = 0.975
    train_mean_losses = metrics.Mean('losses')
    val_mean_losses = metrics.Mean(name='val_losses')
    training_metrics = [metrics.BinaryAccuracy(name='acc') for _ in range(len(l_losses))]
    val_metrics = [metrics.BinaryAccuracy(name='val_acc') for _ in range(len(l_losses))]

    best_val = 0.
    for epoch in range(EPOCHS):
        # Reset state
        reset_states()
        # Training
        if epoch > 5 and epoch % 3 == 0:
            lr = lr_decay * lr
        optimizer = optimizers.Adam(lr)
        print("\n===============================================================")
        print(f"Epoch: {epoch}")
        training_fn = create_training_step(model, l_losses, training_metrics, optimizer)
        calc_loop(train_ds, training_fn, train_mean_losses, training_metrics)
        # Validation
        validate_fn = create_validate_step(model, l_losses, val_metrics)
        val_loss, val_acc = calc_loop(train_ds, training_fn, train_mean_losses, training_metrics, mode='val')
        # Update weight
        if val_acc > best_val:
            best_val = val_acc
            model.save_weights(f'${args["output_dir"]}/checkpoint')
