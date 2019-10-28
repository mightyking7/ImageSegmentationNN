import os
import random
from PIL import Image
from .utils import *

def getIds(dir):
    """
    Returns list of ids in directory
    :arg dir: path to img dir
    :return: list of image file names without extension
    """
    return (i[:-4] for i in os.listdir(dir))

def splitTrainVal(dataset, valPercent=0.05):
    """
    Splits dataset into training images and testing images
    with the given valPercent for percentage of testing images.
    
    :arg valPercent: percentage of dataset to set aside for testing
    :returns: dictionary with training and testing dataset
    """
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * valPercent)

    # shuffle dataset for randomness
    random.shuffle(dataset)
    return {"train": dataset[:-n], "val": dataset[-n:]}

def toCroppedImages(ids, dirImg, suffix, scale):
    """
    Generator the returns a cropped image for each image
    in dirImg identified with an id.
    :param ids: list of image ids to crop
    :param dirImg: path to image folder
    :param suffix: image extension
    :param scale: scale to crop
    :return: cropped square of image
    """

    for id in ids:
        im = resizeImg(Image.open(dirImg + id + suffix), scale=scale)
        im = cropImg(im)
        yield getSquare(im)


def getImageMasks(ids, dirImg, dirMask, scale):
    """
    Transposes img from HWC to CWH, normalizes image pixels,
    and then crops image to adjust for lack of padding.

    :param ids: list of img ids
    :param dirImg: path to img dir
    :param dirMask: path to dir with img masks
    :param scale: scale to resize H,W dimensions of image
    :return: zip object with normalized images, img masks pairs
    """

    # crop images
    imgs = toCroppedImages(ids, dirImg, "bmp", scale)

    # transform from H,W,C to C,W,H
    transposedImgs = map(transposeImgs, imgs)
    normalizedImgs = map(normalizeImgs, transposedImgs)

    # crop masks
    masks = toCroppedImages(ids, dirMask, "bmp", scale)

    return zip(normalizedImgs, masks)
