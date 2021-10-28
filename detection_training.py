import argparse
import time
import os
from tensorflow.keras import metrics, optimizers
from detection.anchor_boxes import LabelEncoder
from detection.ssd import create_ssd_model, create_training_fn, calc_loop, create_val_fn
from utils.data_generator import DetectionGenerator
from utils.dataframe import read_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train_csv', type=str)
    parser.add_argument('--val_csv', type=str)
    parser.add_argument('--image_dir', type=str, default='images')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--output_dir', type=str, default='model')
    args = vars(parser.parse_args())
    return args


def reset_states():
    global train_total_losses
    global train_loc_losses
    global train_classify_losses
    global val_total_losses
    global val_loc_losses
    global val_classify_losses
    train_total_losses.reset_states()
    train_loc_losses.reset_states()
    train_classify_losses.reset_states()
    val_total_losses.reset_states()
    val_loc_losses.reset_states()
    val_classify_losses.reset_states()


if __name__ == '__main__':
    # Settings
    args = parse_args()
    print(args)
    if not os.path.exists(args['output_dir']):
        os.mkdir(args['output_dir'])
    images_dir = args['image_dir']
    # Prepare data
    label_encoder = LabelEncoder()
    train_image_info = read_csv(args["train_csv"], mode='detect')
    val_image_info = read_csv(args["val_csv"], mode='detect')
    train_ds = DetectionGenerator(train_image_info, images_dir, label_encoder=label_encoder, training=True)
    val_ds = DetectionGenerator(val_image_info, images_dir, label_encoder=label_encoder, training=True)
    # Create Model
    ssd_model = create_ssd_model()
    EPOCHS = args['epochs']
    lr = args['lr']
    lr_decay = 0.97
    # Model metrics
    train_total_losses = metrics.Mean('total_losses')
    train_classify_losses = metrics.Mean('classify_losses')
    train_loc_losses = metrics.Mean('loc_losses')

    val_total_losses = metrics.Mean('val_total_losses')
    val_loc_losses = metrics.Mean('val_loc_losses')
    val_classify_losses = metrics.Mean('val_classify_losses')

    best_val = 0.
    for epoch in range(EPOCHS):
        # Reset state
        reset_states()
        # Training
        if epoch > 8 and epoch % 3 == 0:
            lr = lr_decay * lr
        optimizer = optimizers.Adam(lr)
        print("\n===============================================================")
        print(f"Epoch: {epoch + 1}")
        start_time = time.time()
        training_fn = create_training_fn(ssd_model, optimizer)
        calc_loop(train_ds, training_fn, train_total_losses,
                  train_classify_losses, train_loc_losses)
        # Validation
        validate_fn = create_val_fn(ssd_model)
        val_loss = calc_loop(val_ds, validate_fn, val_total_losses,
                             val_classify_losses, val_loc_losses, mode='val')
        # Update weight
        if val_loss > best_val:
            best_val = val_loss
            ssd_model.save_weights(f'{args["output_dir"]}/checkpoint')
        end_time = time.time()
        print(f"After {end_time - start_time}s")
