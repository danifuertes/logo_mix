import os
import time
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Progbar
from PIL import ImageDraw, ImageFont, Image

from utils.utils import get_lines, get_anchors, get_classes
from options import get_options
from yolo3.yolo import YOLO


# Constants
TEST_PHASE = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(opts):

    K.clear_session()

    # Set testing mode (dropout/batchnormalization)
    K.set_learning_phase(TEST_PHASE)

    # Create folders to save outputs (for mAP calculation)
    model_dir = opts.save_dir.split('/')[-1] + '_test_{}'.format(time.strftime("%Y%m%dT%H%M%S"))
    det_dir = os.path.join('./mAP/predictions/', model_dir, 'detection-results')
    gt_dir = os.path.join('./mAP/predictions/', model_dir, 'ground-truth')
    img_dir = os.path.join('./mAP/predictions/', model_dir, 'images')
    if not os.path.exists(det_dir):
        os.makedirs(det_dir)
    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    if not os.path.exists(img_dir) and opts.save_image:
        os.makedirs(img_dir)

    # Load file containing the path of the training images
    lines_test = get_lines(opts.test_path, opts.dataset_path)

    # Input image shape
    input_shape = (opts.image_width, opts.image_height)

    # Get classes and anchors
    class_names = get_classes(opts.classes_path)
    anchors = get_anchors(opts.anchors_path, new_anchors=False, use_bb=opts.use_bb)

    # Load YoloV3 model
    threshold = opts.iou_threshold if opts.use_bb else opts.dist_threshold / max(opts.image_width, opts.image_height)
    weights_path = os.path.join(opts.save_dir, opts.weights_path)
    yolo = YOLO(weights_path, class_names, anchors=anchors, image_shape=input_shape, use_bb=opts.use_bb, iou=threshold,
                score=opts.score_threshold)

    # Prepare progress bar
    steps = len(lines_test)
    progbar = Progbar(target=steps)

    # Counter to calculate mean computation time
    count_time = 0

    # Iterate over each of the images
    for i in range(len(lines_test)):

        split = lines_test[i].split(' ')
        path, gt = split[0], split[1:]

        # Save the ground-truth on a txt (one txt per image)
        filename = path.replace("/", "_").replace('.jpg', ".txt").replace('.png', ".txt")
        results = open(os.path.join(gt_dir, filename), 'w')
        for g in gt:
            marker = g.split(',')
            if len(class_names) > 1:
                results.write(' '.join([class_names[int(marker[-1])], *marker[:-1]]) + '\n')
            else:
                results.write(' '.join([class_names[0], *marker]) + '\n')
        results.close()

        # Load image
        try:
            img = Image.open(path).convert('RGB')
        except:
            print('Open Error! Try again!')
            continue

        # Make predictions
        predictions, t = yolo.detections(img, path, use_bb=opts.use_bb)
        count_time += t

        # Save the results on a txt (one txt per image)
        results = open(os.path.join(det_dir, filename), 'w')

        # Plot predictions
        if opts.show_pred or opts.save_image:
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                      size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32'))
            thickness = (img.size[0] + img.size[1]) // 300
            draw = ImageDraw.Draw(img)

        # For each prediction
        for j in range(predictions.shape[0]):

            if predictions.shape[1] > 1:

                # Bounding boxes
                if opts.use_bb:

                    # Print and save results
                    pred, left, top, right, bottom = predictions[j]
                    label, confidence = pred.split()
                    print('\tClass = {}, Confidence = {}, Xmin = {}, Ymin = {}, Xmax = {}, Ymax = {}' .format(
                        label, confidence, left, top, right, bottom
                    ))
                    results.write(pred + ' ' + left + ' ' + top + ' ' + right + ' ' + bottom + '\n')

                    # Plot prediction
                    if opts.show_pred or opts.save_image:
                        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
                        label_size = draw.textsize(pred, font)
                        if top - label_size[1] >= 0:
                            text_origin = np.array([left, top - label_size[1]])
                        else:
                            text_origin = np.array([left, top + 1])
                        for k in range(thickness):
                            draw.rectangle([left + k, top + k, right - k, bottom - k], outline=(0, 255, 0))

                        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(0, 255, 0))
                        draw.text(text_origin, predictions[j, 0], fill=(0, 0, 0), font=font)

                # Point-based detections
                else:

                    # Print and save results
                    pred, x, y = predictions[j]
                    label, confidence = pred.split()
                    print('\tClass = {}, Confidence = {}, X = {}, Y = {}'.format(label, confidence, x, y))
                    results.write(pred + ' ' + x + ' ' + y + ' ' + '\n')

                    # Plot prediction
                    if opts.show_pred or opts.save_image:
                        x, y = int(x), int(y)
                        label_size = draw.textsize(pred, font)
                        if y - label_size[1] >= 0:
                            text_origin = np.array([x, y - label_size[1]])
                        else:
                            text_origin = np.array([x, y + 1])
                        r = 10
                        draw.ellipse([x - r, y - r, x + r, y + r], fill=(0, 255, 0))
                        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(0, 255, 0))
                        draw.text(text_origin, pred, fill=(0, 0, 0), font=font)

            else:
                results.write(predictions[j, 0])
        results.close()

        # Insert the results on the jpg
        if opts.save_image:
            img.save(os.path.join(img_dir, filename), opts.extension.replace('.', '').upper())
            if not opts.show_pred:
                del draw
        elif opts.show_pred:
            plt.imshow(img)
            plt.show()
            del draw

        # Update progress bar
        progbar.update(i + 1), print('\t')
    print("Mean computation time = {} seconds".format(count_time / steps))
    yolo.close_session()


if __name__ == "__main__":
    main(get_options())
