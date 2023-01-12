import random
import numpy as np
from PIL import Image

from utils.utils import get_data
from yolo3.model import preprocess_true_boxes
from utils.data_aug import data_augmentation, pil2numpy


def random_bb(bb, use_bb=True, shape_objects=None):
    bb = bb[random.randrange(bb.shape[0]), :]
    if use_bb:
        return bb[0], bb[1], bb[2], bb[3], bb[4]
    else:
        xmin = bb[0] - shape_objects[0] // 2
        ymin = bb[1] - shape_objects[1] // 2
        xmax = bb[0] + shape_objects[0] // 2
        ymax = bb[1] + shape_objects[1] // 2
        return xmin, ymin, xmax, ymax, bb[2]


def data_generator(lines, batch_size, input_shape, num_classes, anchors=None, use_bb=True, data_aug=False,
                   use_logomix=False, logomix_perc=0., use_attentive=True, fake_box=None):

    if len(lines) == 0 or batch_size <= 0:
        return None
    input_shape = np.array(input_shape)
    if fake_box is not None:
        fake_box = np.array(fake_box)

    n, i = len(lines), 0
    while True:
        x, y = [], []
        for b in range(batch_size):

            # Get image1 and box1
            if i == 0:
                np.random.shuffle(lines)
            image1, box1 = get_data(lines[i], input_shape, use_bb=use_bb)
            real_box1 = box1[~np.all(box1 == 0, axis=1)]

            # LogoMix
            if use_logomix and real_box1.shape[0] < 20:

                # Convert single-points into fake boxes
                if not use_bb:
                    coord_min = real_box1[..., :2] - fake_box // 2
                    coord_min[coord_min < 0] = 0
                    coord_max = real_box1[..., :2] + fake_box // 2
                    coord_max[coord_max[:, 0] > input_shape[0]] = input_shape[0]
                    coord_max[coord_max[:, 1] > input_shape[1]] = input_shape[1]
                    ax = 0 if len(real_box1[..., 2]) == 1 else 1
                    real_box1 = np.concatenate((coord_min, coord_max, np.expand_dims(real_box1[..., 2], axis=ax)),
                                               axis=1)

                # Get second image
                ind = random.randrange(1, n)
                image2, box2 = get_data(lines[ind], input_shape, use_bb=use_bb)
                real_box2 = box2[~np.all(box2 == 0, axis=1)]

                # Randomly select bounding box of image2. If none bounding box is found, do not insert crop.
                # Make sure width and height are even numbers
                if real_box2.size > 0:

                    # Get crop
                    xmin, ymin, xmax, ymax, category = random_bb(real_box2, use_bb, fake_box)
                    xmax = ((xmax - xmin) // 2) * 2 + xmin
                    ymax = ((ymax - ymin) // 2) * 2 + ymin
                    crop = image2.crop((xmin, ymin, xmax, ymax))
                    crop_w, crop_h = crop.size

                    # If there are not boxes on image1, insert crop in any place
                    if real_box1.size == 0:
                        pos_x = np.random.randint(0, input_shape[0])
                        pos_y = np.random.randint(0, input_shape[1])

                    else:

                        # LogoMix percentage = 0 ==> Crop must not touch any box of image1
                        if logomix_perc == 0:
                            mask = np.ones(np.array(image1.size) + [crop_h, crop_w])
                            for box in real_box1:
                                mask[max(0, box[1] - crop_h):box[3], max(0, box[0] - crop_w):box[2]] = 0

                        # LogoMix percentage > 0 ==> Search where crop can be placed such that overlapping is fulfilled
                        else:

                            # We can assume that crops can be placed partially outside of the image
                            mask = np.zeros(np.array(image1.size).T + [crop_h, crop_w])
                            cutmix_w = (logomix_perc * (real_box1[:, 2] - real_box1[:, 0])).astype(int)
                            cutmix_h = (logomix_perc * (real_box1[:, 3] - real_box1[:, 1])).astype(int)
                            x1 = (real_box1[:, 0] + cutmix_w - crop_w / 2).astype(int)
                            y1 = (real_box1[:, 1] + cutmix_h - crop_h / 2).astype(int)
                            x2 = (real_box1[:, 2] - cutmix_w + crop_w / 2).astype(int)
                            y2 = (real_box1[:, 3] - cutmix_h + crop_h / 2).astype(int)
                            for j in range(real_box1.shape[0]):

                                # Does it fit on the left side?
                                if x1[j] > 0:
                                    mask[max(y1[j], 0):min(y2[j], mask.shape[0] - crop_h), x1[j]] = 1

                                # Does it fit on the top side?
                                if y1[j] > 0:
                                    mask[y1[j], max(x1[j], 0):min(x2[j], mask.shape[1] - crop_w)] = 1

                                # Does it fit on the right side?
                                if x2[j] < mask.shape[1] - crop_w:
                                    mask[max(y1[j], 0):min(y2[j], mask.shape[0] - crop_h), x2[j]] = 1

                                # Does it fit on the bottom side?
                                if y2[j] < mask.shape[0] - crop_h:
                                    mask[y2[j], max(x1[j], 0):min(x2[j], mask.shape[1] - crop_w)] = 1

                            # If Attentive LogoMix, crop overlap = percent with 1 box and <= percent with other boxes
                            if use_attentive:
                                for j in range(real_box1.shape[0]):
                                    mask[y1[j] + 1:y2[j] - 1, x1[j] + 1:x2[j] - 1] = 0

                        # Get a position randomly from the mask
                        if np.max(mask) == 1:
                            positions = np.array(np.where(mask == 1))
                            rand_position = np.random.choice(range(positions.shape[1]))
                            pos_x = positions[1, rand_position]
                            pos_y = positions[0, rand_position]
                            pos_x -= crop_w // 2
                            pos_y -= crop_h // 2
                        else:
                            pos_x = -1
                            pos_y = -1

                    # Place crop on image1 and save its coordinates in box1
                    if pos_x >= 0 and pos_y >= 0:
                        image2 = Image.new('RGB', tuple(np.array(image1.size) + [crop_w, crop_h]))
                        image2.paste(image1, (crop_w//2, crop_h//2))
                        image1 = image2.crop((crop_w/2, crop_h/2, crop_w/2 + image1.size[0], crop_h/2 + image1.size[1]))
                        image1.paste(crop, (pos_x, pos_y))
                        if use_bb:
                            xmin = max(0, pos_x)
                            ymin = max(0, pos_y)
                            xmax = min(input_shape[0], pos_x + crop_w)
                            ymax = min(input_shape[1], pos_y + crop_h)
                            box1[real_box1.shape[0], :] = [xmin, ymin, xmax, ymax, category]
                        else:
                            box1[real_box1.shape[0], :] = [pos_x + crop_w / 2, pos_y + crop_h / 2, category]

            # Transform image
            if data_aug:
                image1, box1 = data_augmentation(image1, box1, use_bb=use_bb)
            image1 = pil2numpy(image1)

            # Append image and its labels to the batch
            x.append(image1)
            y.append(box1)
            i = (i + 1) % n

        # Batch
        x = np.array(x)
        y = np.array(y)
        y_true = preprocess_true_boxes(y, input_shape, anchors, num_classes, use_bb)
        yield [x, *y_true], np.zeros(batch_size)
