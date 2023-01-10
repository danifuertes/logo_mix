import os
import colorsys
import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import load_model
from keras.utils import multi_gpu_model
from timeit import default_timer as timer

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body, yolo_eval_single_point, yolo_body_single_point
from utils.utils import letterbox_image
from utils.data_aug import pil2numpy


class YOLO(object):

    def __init__(self, model_path, class_names, anchors=None, image_shape=(448, 448), use_bb=True, iou=0.5, score=0.3,
                 num_gpu=1):

        # Model info
        self.model_path = model_path
        self.class_names = class_names
        self.anchors = anchors
        self.use_bb = use_bb

        # Number of GPUs
        self.num_gpu = num_gpu

        # Thresholds
        self.iou = iou
        self.score = score

        # Input shape
        self.input_shape = K.placeholder(shape=(2, ))
        self.image_shape = image_shape

        # Load model
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255), 128), self.colors))
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.

    def generate(self):
        """Create YoloV3 model and load weights."""

        # Load model, or construct model and load weights.
        model_path = os.path.expanduser(self.model_path)
        num_classes = len(self.class_names)
        if self.use_bb:
            num_anchors = len(self.anchors)
            tiny_yolo = num_anchors == 6  # default setting
            try:
                self.model = load_model(model_path, compile=False)
            except:
                self.model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                    if tiny_yolo else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
                self.model.load_weights(self.model_path)  # make sure model, anchors and classes match
            else:
                assert self.model.layers[-1].output_shape[-1] == \
                       num_anchors / len(self.model.output) * (num_classes + 5), \
                    'Mismatch between model and given anchor and class sizes'
            print('{} model, {} anchors, and {} classes loaded.'.format(model_path, num_anchors, num_classes))
        else:
            try:
                self.model = load_model(model_path, compile=False)
            except:
                self.model = yolo_body_single_point(Input(shape=(None, None, 3)), num_classes)
                self.model.load_weights(self.model_path)  # make sure model, anchors and classes match
            else:
                assert self.model.layers[-1].output_shape[-1] == \
                       1 / len(self.model.output) * (num_classes + 3), \
                    'Mismatch between model and given anchor and class sizes'
            print('{} model and {} classes loaded.'.format(model_path, num_classes))

        # Generate output tensor targets for filtered bounding boxes.
        if self.num_gpu >= 2:
            self.model = multi_gpu_model(self.model, gpus=self.num_gpu)

        if self.use_bb:
            boxes, scores, classes = yolo_eval(self.model.output, self.anchors, len(self.class_names),
                                               self.input_shape, score_threshold=self.score, iou_threshold=self.iou)
        else:
            boxes, scores, classes = yolo_eval_single_point(self.model.output, len(self.class_names), self.input_shape,
                                                            score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detections(self, image, path, use_bb=True):
        """Make detections for a given image."""

        # Prepare image
        if self.image_shape != (None, None):
            assert self.image_shape[0] % 32 == 0 and self.image_shape[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.image_shape)))
        else:
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.expand_dims(pil2numpy(boxed_image), 0)  # Add batch dimension.

        # Make predictions
        start = timer()
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.model.input: image_data,
                self.input_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        end = timer()
        time = end - start
        print('\nFound {} boxes on {} in {} seconds'.format(len(out_boxes), path, round(time, 4)))

        # Return predictions in a list
        prediction = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            if use_bb:
                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                prediction.append([label, left, top, right, bottom])
            else:
                y, x = box
                y = min(image.size[1], max(0, np.floor(y + 0.5).astype('int32')))
                x = min(image.size[0], max(0, np.floor(x + 0.5).astype('int32')))
                prediction.append([label, x, y])
        return np.asarray(prediction), time

    def close_session(self):
        self.sess.close()
