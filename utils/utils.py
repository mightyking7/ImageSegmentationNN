import numpy as np

def transposeImgs(img):
    """
    transposes numpy array
    :param img: 3 dimensional numpy array with image pixels
    :return: transpose of matrix
    """
    return np.transpose(img, axes=[2,0,1])

def normalizeImgs(x):
    """
    :param x: 3 dimensional numpy array with image pixels
    :return: normalized pixel values
    """
    return x / 255

def getSquare(img):
    """
    :param img: numpy array of pixel values
    :return: tile of img
    """
    h = img.shape[0]
    return img[:, :h]

def resizeImg(img, scale):
    """
    Increases w,h of a PIL image by scale
    :param img: PIL Img
    :param scale: dimension increase
    :return: PIL image resized by scale
    """
    w = img.size[0]
    h = img.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    return img.resize((newW, newH))

def cropImg(img, height=None):
    """
    Crops image with given height
    :param img: PIL image
    :param height: crop y limit
    :return:
    """

    w = img.size[0]
    h = img.size[1]

    if not height:
        diff = 0
    else:
        diff = h - height

    img = img.crop((0, diff // 2, w, h - diff // 2))
    return np.array(img, dtype=np.float32)

