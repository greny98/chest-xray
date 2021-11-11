import argparse
import os
from tensorflow.keras import optimizers, callbacks
from detection.anchor_boxes import LabelEncoder
from detection.ssd import create_ssd_model
from detection.losses import RetinaNetLoss
from utils.data_generator import DetectionGenerator
from siim.data_generator import create_images_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--csv_path', type=str, default='data/siim/train_val.csv')
    parser.add_argument('--image_dir', type=str, default='data/siim/images')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--output_dir', type=str, default='model')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--basenet_ckpt', type=str, default=None)
    args = vars(parser.parse_args())
    return args


# Learning Rate Schedule
def schedule(e, lr):
    if e <= 5 and e % 3 != 0:
        return lr
    return 0.975 * lr


if __name__ == '__main__':
    # Settings
    args = parse_args()
    print(args)
    if not os.path.exists(args['output_dir']):
        os.mkdir(args['output_dir'])
    images_dir = args['image_dir']
    # Prepare data
    label_encoder = LabelEncoder()
    image_info = create_images_info(args["csv_path"])
    images = list(image_info.keys())
    n_train = int(0.9 * len(images))
    train_image_info = {key: image_info[key] for key in images[:n_train]}
    val_image_info = {key: image_info[key] for key in images[n_train:]}
    train_ds = DetectionGenerator(train_image_info, images_dir, label_encoder=label_encoder, training=True,
                                  batch_size=args["batch_size"])
    val_ds = DetectionGenerator(val_image_info, images_dir, label_encoder=label_encoder, training=False,
                                batch_size=args["batch_size"])
    # Create Model
    ssd_model = create_ssd_model(args['basenet_ckpt'])
    # ssd_model.load_weights("").expect_partial()
    loss_fn = RetinaNetLoss(num_classes=1)
    ssd_model.compile(
        loss=loss_fn,
        optimizer=optimizers.Adam(learning_rate=args['lr']))

    ckpt_cb = callbacks.ModelCheckpoint(
        filepath=f"{args['output_dir']}/checkpoint",
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss')
    lr_schedule_cb = callbacks.LearningRateScheduler(schedule)
    # Tensorboard
    tensorboard_cb = callbacks.TensorBoard(log_dir=args['log_dir'])
    ssd_model.fit(train_ds,
                  validation_data=val_ds,
                  callbacks=[ckpt_cb, lr_schedule_cb, tensorboard_cb],
                  epochs=args['epochs'])
