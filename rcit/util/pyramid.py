import cv2
import numpy as np


def pyramid_down(img, levels):
    """

    :param img: first radar image
    :param levels: No. of labels in the pyramid
    :return: pyramid
    """

    # Initializing pyramid
    pyr = np.zeros((img.shape[0], img.shape[1], levels))

    # updating first real size radar image at first position
    pyr[:, :, 0] = img
    shapes = [[img.shape[0], img.shape[1]]]

    # updating other values
    for i in range(1, levels):
        temp = cv2.pyrDown(img)
        pyr[0:temp.shape[0], 0:temp.shape[1], i] = temp
        shapes.append([temp.shape[0], temp.shape[1]])
        img = temp

    return pyr, shapes
