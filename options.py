import os
import random
import argparse
import numpy as np
import tensorflow as tf


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed):
    """Set seed"""
    if seed is None:
        seed = random.randrange(100)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    tf.set_random_seed(seed)  # Tensorflow1 module
    # tf.random.set_seed(seed)  # Tensorflow2 module


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Yolov3 trained with LogoMix (data augmentation technique for logo detection).")

    # SEED
    parser.add_argument('--seed', type=int, default=10101, help="Seed")

    # SAVE
    parser.add_argument('--save_dir', type=str, default='models', help="Directory to save models")
    parser.add_argument('--restore_model', type=str2bool, default=False, help="True to restore a model for training")
    parser.add_argument('--weights_path', type=str, default='yolo3/yolo_weights.h5',
                        help='Pretrained weights file. None for random weights')

    # DATASET
    parser.add_argument('--dataset_path', type=str, default='/path/to/dataset',
                        help='Directory containing the dataset.')
    parser.add_argument('--dataset_name', type=str, default='dataset_name',
                        help='Directory containing the dataset.')
    parser.add_argument('--train_path', type=str, default='train.txt',
                        help='Txt with train files')
    parser.add_argument('--test_path', type=str, default='test.txt',
                        help='Txt with test files')
    parser.add_argument('--val_path', type=str, default='val.txt',
                        help='Txt with validation files. If None, val set is created from train set using val_split')
    parser.add_argument('--val_perc', type=float, default=0.1, help='Percentage of files from the train set used to'
                                                                    'validate. Used if val_path is None or not exists')
    parser.add_argument('--extension', type=str, default='.jpg', help='Image extension.')

    # Classes
    parser.add_argument('--classes_path', type=str, default='classes.txt',
                        help='Txt with the classes.')

    # Anchors
    parser.add_argument('--new_anchors', type=str2bool, default=False,
                        help='True to calculate new anchors and save them in anchors_path')
    parser.add_argument('--anchors_path', type=str, default='yolo3/yolo_anchors.txt', help='Txt with the anchors.')
    parser.add_argument('--num_clusters', type=int, default=9,
                        help='Number of clusters of K-Means for the anchors. Anchors will be calculated only if'
                             'new_anchors=True. They will be saved in anchors_path file')

    # TRAIN PARAMETERS
    parser.add_argument('--train_frozen', type=str2bool, default=True, help="Freeze almost all layers while training")
    parser.add_argument('--train_unfrozen', type=str2bool, default=True, help="Unfreeze all layers while training")

    # Batch size
    parser.add_argument('--batch_frozen', type=int, default=32, help='Batch size during the frozen stage')
    parser.add_argument('--batch_unfrozen', type=int, default=16, help='Batch size during the unfrozen stage')

    # Learning rate
    parser.add_argument('--lr_frozen', type=float, default=1e-3, help='Initial learning rate for the frozen stage')
    parser.add_argument('--lr_unfrozen', type=float, default=1e-4, help='Initial learning rate for the unfrozen stage')

    # Epochs
    parser.add_argument('--epochs_frozen', type=int, default=100, help='Number of epochs for the frozen stage')
    parser.add_argument('--epochs_unfrozen', type=int, default=300, help='Number of epochs for the unfrozen stage')
    parser.add_argument('--initial_epoch', type=int, default=0, help='Initial epoch')

    # Image size
    parser.add_argument('-iw', '--image_width', type=int, default=448, help="Image width")
    parser.add_argument('-ih', '--image_height', type=int, default=448, help="Image height")

    # EarlyStop and ReduceOnPlateau
    parser.add_argument('--early_stop', type=str2bool, default=False,
                        help="Stop the training if loss is not improved in less than early_stop_period epochs")
    parser.add_argument('--early_stop_period', type=int, default=100,
                        help='Training stops when model does not improve during this number of consecutive epochs')
    parser.add_argument('--reduce_lr_period', type=int, default=30,
                        help='The lr is reduced if loss is not improved in less than reduce_lr_period epochs')

    # Data augmentation
    parser.add_argument('--data_aug', type=str2bool, default=True,
                        help="Apply random transformations to the image during training")
    parser.add_argument('--crop_aug', type=str2bool, default=True,
                        help="Apply random transformations to the new crop inserted during LogoMix training")

    # BOUNDING BOXES / POINT-BASED ANNOTATIONS
    parser.add_argument('--use_bb', type=str2bool, default=True,
                        help="True to use bounding boxes. False to use point-based annotations")
    parser.add_argument('-fw', '--fake_width', type=int, default=70, help="Fake box width for point-based detections")
    parser.add_argument('-fh', '--fake_height', type=int, default=70, help="Fake box height for point-based detections")
    # Flickr = (70,70) | OpenLogo = (60,40)

    # LOGOMIX
    parser.add_argument('--use_logomix', type=str2bool, default=True, help="Use LogoMix training strategy")
    parser.add_argument('--use_attentive', type=str2bool, default=False,
                        help="True to use Attentive LogoMix. False to use Non-Attentive LogoMix")
    parser.add_argument('--logomix_perc', type=float, default=0.15,
                        help="LogoMix percentage of overlapping between 2 bounding boxes. Must be in the range [0, 1].")

    # Prediction
    parser.add_argument('-s_th', '--score_threshold', type=float, default=0.3, help="Score threshold for predictions")
    parser.add_argument('-i_th', '--iou_threshold', type=float, default=0.5,
                        help="IoU threshold for bounding box predictions")
    parser.add_argument('-d_th', '--dist_threshold', type=float, default=16,
                        help="Distance threshold for point-based detections")

    # Show images
    parser.add_argument('--show_pred', type=str2bool, default=False, help="True to show the predicted images")
    # parser.add_argument('--show_gt', type=str2bool, default=False, help="True to show the annotations in the images")
    parser.add_argument('--save_image', type=str2bool, default=False, help="True to save the predicted images")

    # GPU
    parser.add_argument("--num_gpu", type=int, default=1, help="Number of GPUs.")

    # OPTIONS
    opts = parser.parse_args(args)
    opts.train_path = os.path.join(opts.dataset_path, opts.dataset_name, opts.train_path)
    opts.test_path = os.path.join(opts.dataset_path, opts.dataset_name, opts.test_path)
    opts.val_path = os.path.join(opts.dataset_path, opts.dataset_name, opts.val_path)
    opts.classes_path = os.path.join(opts.dataset_path, opts.dataset_name, opts.classes_path)

    # Check everything is ok
    assert opts.image_width > 0 and opts.image_height > 0, "img_width and img_height must be positive integers"
    assert opts.image_width % 32 == 0 and opts.image_height % 32 == 0, "Multiples of 32 required for the image shape"
    assert opts.batch_frozen > 0 or opts.lr_frozen > 0 or opts.batch_unfrozen > 0 or opts.lr_unfrozen > 0,\
        "Batch size and learning rate must be positive numbers"
    assert 0 < opts.score_threshold <= 1, "score_threshold must be in the range [0, 1]"
    assert 0 < opts.iou_threshold <= 1, "iou_threshold must be in the range [0, 1]"
    assert 0 < opts.dist_threshold <= np.min([opts.image_width, opts.image_height]), \
        "distance_threshold must be in the range [0, max(img_width, img_height)]"
    assert os.path.isdir(opts.dataset_path), "dataset_path does not exist"
    assert os.path.isfile(opts.train_path), "train_path does not exist"
    assert os.path.isfile(opts.test_path), "test_path does not exist"
    assert os.path.isfile(opts.val_path) or 0 <= opts.val_perc <= 1, \
        "Either val_path must exist or val_perc must be in the range [0, 1]"
    assert 0 <= opts.logomix_perc <= 1, "logomix_perc must be in the range [0, 1]"
    assert (opts.restore_model and os.path.isfile(os.path.join(opts.save_dir, opts.weights_path))) or \
           not opts.restore_model, "Weights not exists"
    assert opts.extension in ['.jpg', '.png', '.gif'], "Extension not in the list ['.jpg', '.png', '.gif']"
    assert opts.num_gpu >= 0, "num_gpu cannot be < 0"

    # Set seed
    set_seed(opts.seed)
    return opts
