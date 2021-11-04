import argparse
import os
from tensorflow.keras import metrics, optimizers, callbacks
from detection.anchor_boxes import LabelEncoder
from detection.ssd import create_ssd_model
from detection.losses import RetinaNetLoss
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
    train_ds = DetectionGenerator(train_image_info, images_dir, label_encoder=label_encoder, training=True,
                                  batch_size=args["batch_size"])
    val_ds = DetectionGenerator(val_image_info, images_dir, label_encoder=label_encoder, training=False,
                                batch_size=args["batch_size"])
    # Create Model
    ssd_model = create_ssd_model()
    # ssd_model.load_weights("").expect_partial()
    loss_fn = RetinaNetLoss()
    ssd_model.compile(
        loss=loss_fn,
        optimizer=optimizers.Adam(learning_rate=args['lr']))

    ckpt_cb = callbacks.ModelCheckpoint(
        filepath=f"{args['output_dir']}/checkpoint",
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss')
    lr_schedule_cb = callbacks.ReduceLROnPlateau(patience=5, factor=0.25, min_lr=1e-6)
    ssd_model.fit(train_ds,
                  validation_data=val_ds,
                  callbacks=[ckpt_cb, lr_schedule_cb],
                  epochs=args['epochs'])
