import cv2
import numpy as np
# import cupy as cp
import scipy.signal as signal

from rcit.util.pre_process.curvature_filter.cf_interface import cf_filter
from rcit.util.pre_process.input_neis import neighbors_eight


def input_filter(returnMat, filter_type):
    # for all nan values in the returnMat, they are all converted to 0
    origin_img = outs = np.nan_to_num(returnMat)

    # choosing filtering methods if the filter type is 'mf', median filter method is applied
    if filter_type == 'mf':
        outs = signal.medfilt2d(outs, (3, 3))

    # if the filter type is 'cf', Gaussian Curvature method is applied
    if filter_type == 'cf':
        outs = cf_filter(outs, 'tv')

    # getting the neighbor pixels for each pixel by method 'neighbors_eight',
    # the number of neighbors is 8
    nei_pixels = neighbors_eight(outs)
    # nei_pixels = pixel_neighbor_identification(outs, 8)
    row_nei, col_nei = nei_pixels.shape
    row_outs, col_outs = outs.shape

    length_with_null_values = np.zeros([row_nei, 1], dtype=int)
    nei_mat_pixel = np.zeros([row_nei, 2], dtype=float)

    index_x = list(range(0, row_nei))
    for i in index_x:
        nan_list = np.argwhere(np.isnan(nei_pixels[i, :]))
        length_with_null_values[i] = len(nan_list)

    nei_mat_pixel[:, 0] = nei_pixels[:, 0]
    nei_mat_pixel[:, 1] = length_with_null_values[:, 0]

    nei_mat_pixel[:, 0][nei_mat_pixel[:, 0] <= 0] = np.nan

    pixels = np.transpose(nei_mat_pixel[:, 0])
    filter_pixel = np.reshape(pixels, (row_outs, col_outs))
    return origin_img, filter_pixel


def isolate_echo_remove(inputimg, n=3, thr=0):
    """Apply a binary morphological opening to filter small isolated echoes.

    Parameters
    ----------
    inputimg : array-like
        Array of shape (m,n) containing the input precipitation field.
    n : int
        The structuring element size [px].
    thr : float
        The rain/no-rain threshold to convert the image into a binary image.

    Returns
    -------
    R : array
        Array of shape (m,n) containing the cleaned precipitation field.
    """

    # convert to binary image (rain/no rain)
    field_bin = np.ndarray.astype(inputimg > thr, "uint8")

    # build a structuring element of size (nx)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))

    # apply morphological opening (i.e. erosion then dilation)
    field_bin_out = cv2.morphologyEx(field_bin, cv2.MORPH_OPEN, kernel)

    # build mask to be applied on the original image
    mask = (field_bin - field_bin_out) > 0

    # filter out small isolated echoes based on mask
    inputimg[mask] = np.nanmin(inputimg)

    return inputimg


def outside_in_fill(image):
    """
    Outside in fill mentioned in paper
    :param image: Image matrix to be filled
    :return: output
    """

    rows, cols = image.shape[:2]

    col_start = 0
    col_end = cols
    row_start = 0
    row_end = rows
    lastValid = np.full([2], np.nan)
    while col_start < col_end or row_start < row_end:
        for c in range(col_start, col_end):
            if not np.isnan(image[row_start, c, 0]):
                lastValid = image[row_start, c, :]
            else:
                image[row_start, c, :] = lastValid

        for r in range(row_start, row_end):
            if not np.isnan(image[r, col_end - 1, 0]):
                lastValid = image[r, col_end - 1, :]
            else:
                image[r, col_end - 1, :] = lastValid

        for c in reversed(range(col_start, col_end)):
            if not np.isnan(image[row_end - 1, c, 0]):
                lastValid = image[row_end - 1, c, :]
            else:
                image[row_end - 1, c, :] = lastValid

        for r in reversed(range(row_start, row_end)):
            if not np.isnan(image[r, col_start, 0]):
                lastValid = image[r, col_start, :]
            else:
                image[r, col_start, :] = lastValid

        if col_start < col_end:
            col_start = col_start + 1
            col_end = col_end - 1

        if row_start < row_end:
            row_start = row_start + 1
            row_end = row_end - 1

    output = image

    return output
