import cv2
import numpy as np
from PIL import Image, ImageEnhance


def bbox_area(bbox):
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def rand(a=0., b=1.):
    return np.random.rand()*(b-a) + a


def numpy2pil(image):
    return Image.fromarray((image * 255).astype('uint8'), 'RGB')


def pil2numpy(image):
    return np.array(image) / 255.


def data_augmentation(image, box, max_boxes=20, use_bb=True, is_crop=False):
    """
    Data augmentation techniques.

    # Arguments
    image: image that will be modified.
    box: label of the given image.
    max_boxes: maximum number of boxes allowed for a given image.
    use_bb: use bounding boxes or punctual annotations.
    is_crop: a crop is augmented instead of an image.
    """

    # Image size
    w, h = image.size

    # Labels
    box_data = np.zeros((max_boxes, 5)) if use_bb else np.zeros((max_boxes, 3))
    box = box[~np.all(box == 0, axis=1)]

    # Resize
    if rand() > 0.5:
        jitter = 0.3
        min_scale_factor = 1 if is_crop else 0.75
        max_scale_factor = 1.25
        new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(min_scale_factor, max_scale_factor)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        if is_crop:
            box[:, [0, 2]] = box[:, [0, 2]] * nw / w
            box[:, [1, 3]] = box[:, [1, 3]] * nh / h
            w = nw
            h = nh
        else:
            dx = int(rand(0, w - nw))
            dy = int(rand(0, h - nh))
            new_image = Image.new('RGB', (w, h), (0, 0, 0))
            new_image.paste(image, (dx, dy))
            image = new_image
            if use_bb:
                box[:, [0, 2]] = box[:, [0, 2]] * nw / w + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / h + dy
            else:
                box[:, 0] = box[:, 0] * nw / w + dx
                box[:, 1] = box[:, 1] * nh / h + dy

    # Horizontal Flip
    if rand() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if not is_crop:
            if use_bb:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            else:
                box[:, 0] = w - box[:, 0]

    # Vertical Flip
    if rand() > 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if not is_crop:
            if use_bb:
                box[:, [1, 3]] = h - box[:, [3, 1]]
            else:
                box[:, 1] = h - box[:, 1]

    # PIL to NumPy
    image = pil2numpy(image)

    # Rotation
    if rand() > 0.5 and use_bb and not is_crop:
        image, box = rotation(image, box)

    # Shear
    if rand() > 0.5 and not is_crop:
        shear_factor = 0.2
        shear_factor = rand(-shear_factor, shear_factor)
        M = np.array([[1, abs(shear_factor), 0], [0, 1, 0]])
        nW = w + abs(shear_factor * h)
        image = cv2.warpAffine(image, M, (int(nW), h))
        image = cv2.resize(image, (w, h))
        scale_factor_x = nW / w
        if use_bb:
            box[:, [0, 2]] += ((box[:, [1, 3]]) * abs(shear_factor)).astype(int)
            box[:, :4] = box[:, :4] / [scale_factor_x, 1, scale_factor_x, 1]

    # NumPy to PIL
    image = numpy2pil(image)

    # Brightness change
    if rand() > 0.5:
        factor = rand(0.75, 1.25)
        image = ImageEnhance.Brightness(image).enhance(factor)

    # Contrast change
    if rand() > 0.5:
        factor = rand(0.75, 1.25)
        image = ImageEnhance.Contrast(image).enhance(factor)

    # Amend box
    box[:, 0:2][box[:, 0:2] < 0] = 0
    if use_bb:
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

        min_box_area = 200
        box = box[bbox_area(box) > min_box_area]
    else:
        box[:, 0][box[:, 0] > w] = w
        box[:, 1][box[:, 1] > h] = h

    box_data[:len(box)] = box
    return image, box_data


def rotation(image, box):

    # Random angle
    angle = rand(-45, 45)

    # Image shape
    w, h = image.shape[1], image.shape[0]
    cx, cy = w // 2, h // 2

    # Rotate image
    img = rotate_im(image, angle)

    # Rotate box
    corners = get_corners(box)
    corners = np.hstack((corners, box[:, 4:]))
    corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)
    new_bbox = get_enclosing_box(corners)

    # Resize image and box
    scale_factor_x = img.shape[1] / w
    scale_factor_y = img.shape[0] / h
    img = cv2.resize(img, (w, h))
    new_bbox[:, :4] = new_bbox[:, :4] / [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]
    bboxes = new_bbox
    bboxes = clip_box(bboxes, [0, 0, w, h], 0.25)

    return img, bboxes


def rotate_im(image, angle):
    """Rotate the image.

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black.

    Parameters
    ----------

    image : numpy.ndarray
        numpy image

    angle : float
        angle by which the image is to be rotated

    Returns
    -------

    numpy.ndarray
        Rotated Image

    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

    #    image = cv2.resize(image, (w,h))
    return image


def rotate_box(corners, angle, cx, cy, h, w):
    """Rotate the bounding box.


    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    angle : float
        angle by which the image is to be rotated

    cx : int
        x coordinate of the center of image (about which the box will be rotated)

    cy : int
        y coordinate of the center of image (about which the box will be rotated)

    h : int
        height of the image

    w : int
        width of the image

    Returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated


def get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box

    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    Returns
    -------

    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final


def clip_box(bbox, clip_box, alpha):
    """Clip the bounding boxes to the borders of an image

    Parameters
    ----------

    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    clip_box: numpy.ndarray
        An array of shape (4,) specifying the diagonal co-ordinates of the image
        The coordinates are represented in the format `x1 y1 x2 y2`

    alpha: float
        If the fraction of a bounding box left in the image after being clipped is
        less than `alpha` the bounding box is dropped.

    Returns
    -------

    numpy.ndarray
        Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the
        number of bounding boxes left are being clipped and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:, 0], clip_box[0]).reshape(-1, 1)
    y_min = np.maximum(bbox[:, 1], clip_box[1]).reshape(-1, 1)
    x_max = np.minimum(bbox[:, 2], clip_box[2]).reshape(-1, 1)
    y_max = np.minimum(bbox[:, 3], clip_box[3]).reshape(-1, 1)

    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:, 4:]))

    delta_area = ((ar_ - bbox_area(bbox)) / ar_)

    mask = (delta_area < (1 - alpha)).astype(int)

    bbox = bbox[mask == 1, :]

    return bbox


def get_corners(bboxes):
    """Get corners of bounding boxes

    Parameters
    ----------

    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    """
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners
