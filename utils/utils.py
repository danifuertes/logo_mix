import os
import pickle
import numpy as np
from PIL import Image
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda
from keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import Callback

from yolo3.model import yolo_body, yolo_body_single_point, tiny_yolo_body, yolo_loss, yolo_loss_single_point
from yolo3.kmeans import kmeans_anchors


def letterbox_image(image, size):
    """Resize image with unchanged aspect ratio using padding."""
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def get_lines(path_to_lines, dataset_path, shuffle=True):
    """Load lines in path_to_lines, add to them the absolute path, and (maybe) shuffle them."""
    with open(path_to_lines) as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if lines[i][0] == '/':
            lines[i] = lines[i][1:]
        lines[i] = os.path.join(dataset_path, lines[i].replace('\n', ''))
    if shuffle:
        np.random.shuffle(lines)
    return lines


def get_classes(classes_path):
    """"
    Loads the classes.
    Arguments:
        classes_path: path to file containing the list of classes.
    Outputs: list of the classes.
    """
    if not os.path.exists(classes_path):
        return ['object']
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path, num_anchors=9, new_anchors=False, use_bb=True):
    """"
    Loads the anchors.
    Arguments:
        anchors_path: path to file containing the list of anchors.
        num_anchors: number of anchors.
        new_anchors: True to calculate new anchors and save them in anchors_path.
        use_bb: True to use bounding boxes. Without bounding boxes, anchors are not necessary.
    Outputs: anchors.
    """
    if use_bb:
        if new_anchors:
            kmeans_anchors(num_anchors, anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)
    return None


def load_weights(model, path):
    """Load weights in path on model."""
    if path is not None and os.path.isfile(path):
        try:
            model.load_weights(path, by_name=True, skip_mismatch=True)
            print("Loaded model from {}".format(path))
        except:
            print("Impossible to find weight path. Returning untrained model")
    else:
        print("Impossible to find weight path. Returning untrained model")
    return model


def get_model(input_shape, num_classes, anchors=None, use_bb=True, restore_model=True, weights_path='', freeze_body=2,
              tiny_yolo=False, num_gpu=1, iou_th=.5):
    """
    Creates the training YoloV3 model.
    Arguments:
        input_shape: shape of the image.
        num_classes: number of classes of the model.
        anchors: numpy array of anchors.
        use_bb: True to use bounding box detections. False to use point-based detections.
        restore_model: if True, weights from a previous training are loaded.
        weights_path: path to the file containing pretrained weights. Used only if restore_model = True.
        freeze_body: freeze all the layers except from the output layers.
        tiny_yolo: True to load Tiny YoloV3 model.
        num_gpu: number of GPUs.
        iou_th: IoU threshold (for boxes) or Distance threshold (for points).
    Outputs:
        model: YOLOv3 model.
    """
    # Get a new session
    K.clear_session()

    # Get tensor sized as indicated by the input shape, number of anchors and number of classes
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape

    # YoloV3 with bounding box detection
    if use_bb and not tiny_yolo:

        # Create the body of the model
        num_anchors = len(anchors)
        y_true = [
            Input(
                shape=(
                    h // {0: 32, 1: 16, 2: 8}[i],
                    w // {0: 32, 1: 16, 2: 8}[i],
                    num_anchors // 3,
                    num_classes + 5
                )
            ) for i in range(3)
        ]
        model_body = yolo_body(image_input, num_anchors // 3, num_classes)
        print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    # YoloV3 with point-based detection
    elif not use_bb and not tiny_yolo:

        # Create the body of the model
        y_true = [
            Input(
                shape=(
                    h // {0: 32, 1: 16, 2: 8}[i],
                    w // {0: 32, 1: 16, 2: 8}[i],
                    num_classes + 3
                )
            ) for i in range(3)
        ]
        model_body = yolo_body_single_point(image_input, num_classes)
        print('Create YOLOv3 model with {} classes.'.format(num_classes))

    # Tiny YoloV3 with bounding box detection
    elif use_bb and tiny_yolo:

        # Create the body of the model
        num_anchors = len(anchors)
        y_true = [
            Input(
                shape=(
                    h // {0: 32, 1: 16}[i],
                    w // {0: 32, 1: 16}[i],
                    num_anchors // 2,
                    num_classes + 5
                )
            ) for i in range(3)
        ]
        model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
        print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    # Tiny YoloV3 with point-based detection
    else:
        NotImplementedError('Tiny YoloV3 is not supported with point-based detections')
        return

    # Load pretrained weights
    if (restore_model or weights_path == 'yolo3/yolo_weights.h5') and os.path.isfile(weights_path):
        model_body = load_weights(model_body, weights_path)
        print('Load weights {}.'.format(weights_path))

        # Freeze darknet53 body or freeze all but 3 output layers.
        if freeze_body in [1, 2]:
            num = (185, len(model_body.layers) - 3)[freeze_body - 1] \
                if not tiny_yolo else (20, len(model_body.layers) - 2)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    # Configure the loss function
    model_loss = Lambda(
        yolo_loss if use_bb else yolo_loss_single_point,
        output_shape=(1,),
        name='yolo_loss' if use_bb else 'yolo_loss_single_point',
        arguments={
            'anchors': anchors,
            'num_classes': num_classes,
            'ignore_thresh': iou_th if not tiny_yolo else 0.7
        }
    )([*model_body.output, *y_true])

    # Return the model
    model = Model([model_body.input, *y_true], model_loss)
    model.summary()
    if num_gpu > 1:
        model = multi_gpu_model(model, gpus=num_gpu)
    return model


def get_data(annotation_line, input_shape, use_bb=True, max_boxes=20):
    """
    Load training data.

    # Arguments
    annotation_line: path to the images.
    input_shape: shape of the images.
    use_bb: use bounding boxes or punctual annotations.
    max_boxes: maximum number of boxes allowed for a given image.

    # Outputs
    new_image: loaded image.
    box_data: labels of the loaded image.
    """

    # Load data
    line = annotation_line.split()
    image = Image.open(line[0])

    # Resize images
    iw, ih = image.size
    h, w = input_shape
    image = image.resize((w, h), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (0, 0))

    box_data = np.zeros((max_boxes, 5)) if use_bb else np.zeros((max_boxes, 3))
    box = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
    if len(box) > 0 and ((use_bb and box.shape[-1] == 4) or ((not use_bb) and box.shape[-1] == 2)):
        box = np.concatenate((box, np.zeros(box.shape)[..., 0, None]), axis=-1)
    if len(box) > 0:
        if use_bb:
            box[:, [0, 2]] = box[:, [0, 2]] * w / iw
            box[:, [1, 3]] = box[:, [1, 3]] * h / ih
        else:
            box[:, 0] = box[:, 0] * w / iw
            box[:, 1] = box[:, 1] * h / ih
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box_data[:len(box)] = box
    return new_image, box_data.astype(int)


class HistoryCallback(Callback):

    def __init__(self, path):
        super().__init__()
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        if os.path.isfile(os.path.join(self.path, 'history.pkl')):
            with open(os.path.join(self.path, 'history.pkl'), 'rb') as f:
                history = pickle.load(f)
            history[epoch] = logs
        else:
            history = {epoch: logs}
        with open(os.path.join(self.path, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)
