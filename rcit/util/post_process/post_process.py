import copy

import cv2
import numpy as np
import scipy
import scipy.signal as signal
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator

from rcit.motion_cal.motion_interpolate.motion_interp_general import motion_interp_linear, motion_interp_spline
from rcit.motion_cal.motion_interpolate.motion_interp_dace import motion_interp_kirgin, motion_interp_kriging_local


def motion_interp_nan(motion):
    mx, my = motion.shape
    uo = motion
    ut = motion_interp_spline(motion)

    nan_indice = np.isnan(motion)
    ut[nan_indice] = np.nanmedian(motion)
    ut = signal.medfilt2d(ut, (3, 3))

    ut[0, 0] = ut[1, 1]
    ut[mx - 1, 0] = ut[mx - 2, 1]
    ut[0, my - 1] = ut[1, my - 2]
    ut[mx - 1, my - 1] = ut[mx - 2, my - 2]

    uo[nan_indice] = ut[nan_indice]
    return uo


def inpaint_nans(input):
    nans = np.isnan(input)
    input_1 = input
    ipn_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])  # kernel for inpaint_nans
    while np.sum(nans) > 0:
        input_1[nans] = 0
        vNeighbors = signal.convolve2d((nans == False), ipn_kernel, mode='same', boundary='symm')
        input_2 = signal.convolve2d(input_1, ipn_kernel, mode='same', boundary='symm')
        input_2[vNeighbors > 0] = input_2[vNeighbors > 0] / vNeighbors[vNeighbors > 0]
        input_2[vNeighbors == 0] = np.nan
        input_2[(nans == False)] = input_1[(nans == False)]
        input_1 = input_2
        nans = np.isnan(input_1)
    return input_2


def motion_interp_option(motion, i_interp):
    uo = motion

    # linear interpolation
    if i_interp == 1:
        uo = motion_interp_linear(motion)

    elif i_interp == 2:
        uo = motion_interp_spline(motion)

    elif i_interp == 3:
        uo = motion_interp_kirgin(motion)

    else:
        uo = motion_interp_kriging_local(motion)

    return uo


def _interpolate_nan(value):
    value_2d = copy.deepcopy(value)

    W = 1 / (2 ** 0.5)
    for j in range(1, value_2d.shape[0] - 1):
        for i in range(1, value_2d.shape[1] - 1):
            if np.isnan(value_2d[j, i]):
                value_2d[j, i] = (value_2d[j + 1, i] + value_2d[j - 1, i] + value_2d[j, i + 1] + value_2d[j, i - 1]) / \
                                 (4 + 4 * W) + (value_2d[j + 1, i + 1] + value_2d[j + 1, i - 1] +
                                                value_2d[j - 1, i + 1] + value_2d[j - 1, i - 1]) * W / (4 + 4 * W)

    for j in range(1, value_2d.shape[0] - 1):
        i = 0
        if np.isnan(value_2d[j, i]):
            value_2d[j, i] = (value_2d[j + 1, i] + value_2d[j - 1, i] + value_2d[j, i + 1]) / (3 + 3 * W) + \
                             (value_2d[j + 1, i + 1] + value_2d[j - 1, i + 1]) * W / (2 + 2 * W)
        i = value_2d.shape[1] - 1
        if np.isnan(value_2d[j, i]):
            value_2d[j, i] = (value_2d[j + 1, i] + value_2d[j - 1, i] + value_2d[j, i - 1]) / (3 + 3 * W) + \
                             (value_2d[j + 1, i - 1] + value_2d[j - 1, i - 1]) * W / (2 + 2 * W)

    for i in range(1, value_2d.shape[1] - 1):
        j = 0
        if np.isnan(value_2d[j, i]):
            value_2d[j, i] = (value_2d[j + 1, i] + value_2d[j, i - 1] + value_2d[j, i + 1]) / (3 + 3 * W) + \
                             (value_2d[j + 1, i + 1] + value_2d[j + 1, i - 1]) * W / (2 + 2 * W)
        j = value_2d.shape[0] - 1
        if np.isnan(value_2d[j, i]):
            value_2d[j, i] = (value_2d[j, i + 1] + value_2d[j - 1, i] + value_2d[j, i - 1]) / (3 + 3 * W) + \
                             (value_2d[j - 1, i + 1] + value_2d[j - 1, i - 1]) * W / (2 + 2 * W)

    if np.isnan(value_2d[0, 0]):
        value_2d[0, 0] = (value_2d[1, 0] + value_2d[0, 1]) / (2 + 2 * W) + value_2d[1, 1] * W / (1 + 1 * W)
    if np.isnan(value_2d[-1, 0]):
        value_2d[-1, 0] = (value_2d[-2, 0] + value_2d[-1, 1]) / (2 + 2 * W) + value_2d[-2, 1] * W / (1 + 1 * W)
    if np.isnan(value_2d[0, -1]):
        value_2d[0, -1] = (value_2d[0, -2] + value_2d[1, -1]) / (2 + 2 * W) + value_2d[1, -2] * W / (1 + 1 * W)
    if np.isnan(value_2d[-1, -1]):
        value_2d[-1, -1] = (value_2d[-1, -2] + value_2d[-2, -1]) / (2 + 2 * W) + value_2d[-2, -2] * W / (1 + 1 * W)

    # print('Number of NAN value : ', end='')
    # print('%d / %d' % (np.count_nonzero(np.isnan(value_2d)), value_2d.size))

    last_nan_count = np.count_nonzero(np.isnan(value_2d))

    while True:
        for j in range(1, value_2d.shape[0] - 1):
            for i in range(1, value_2d.shape[1] - 1):
                if np.isnan(value_2d[j, i]):
                    value_2d[j, i] = np.nanmean(
                        [value_2d[j + 1, i], value_2d[j - 1, i], value_2d[j, i + 1], value_2d[j, i - 1],
                         value_2d[j + 1, i + 1], value_2d[j + 1, i - 1], value_2d[j - 1, i + 1],
                         value_2d[j - 1, i - 1]])
        for i in range(1, value_2d.shape[1] - 1):
            j = 0
            if np.isnan(value_2d[j, i]):
                value_2d[j, i] = np.nanmean(
                    [value_2d[j + 1, i], value_2d[j, i + 1], value_2d[j, i - 1], value_2d[j + 1, i + 1],
                     value_2d[j + 1, i - 1]])
            j = value_2d.shape[0] - 1
            if np.isnan(value_2d[j, i]):
                value_2d[j, i] = np.nanmean(
                    [value_2d[j - 1, i], value_2d[j, i + 1], value_2d[j, i - 1], value_2d[j - 1, i + 1],
                     value_2d[j - 1, i - 1]])

        for j in range(1, value_2d.shape[0] - 1):
            i = 0
            if np.isnan(value_2d[j, i]):
                value_2d[j, i] = np.nanmean(
                    [value_2d[j + 1, i], value_2d[j - 1, i], value_2d[j, i + 1], value_2d[j + 1, i + 1],
                     value_2d[j - 1, i + 1]])
            i = value_2d.shape[1] - 1
            if np.isnan(value_2d[j, i]):
                value_2d[j, i] = np.nanmean(
                    [value_2d[j + 1, i], value_2d[j - 1, i], value_2d[j, i - 1], value_2d[j + 1, i - 1],
                     value_2d[j - 1, i - 1]])

        if np.isnan(value_2d[0, 0]):
            value_2d[0, 0] = np.nanmean([value_2d[1, 0], value_2d[0, 1], value_2d[1, 1]])
        if np.isnan(value_2d[-1, 0]):
            value_2d[-1, 0] = np.nanmean([value_2d[-2, 0], value_2d[-1, 1], value_2d[-2, 1]])
        if np.isnan(value_2d[0, -1]):
            value_2d[0, -1] = np.nanmean([value_2d[0, -2], value_2d[1, -1], value_2d[1, -2]])
        if np.isnan(value_2d[-1, -1]):
            value_2d[-1, -1] = np.nanmean([value_2d[-1, -2], value_2d[-2, -1], value_2d[-2, -2]])

        if np.count_nonzero(np.isnan(value_2d)) == 0 or np.count_nonzero(np.isnan(value_2d)) == last_nan_count:
            break
        else:
            last_nan_count = np.count_nonzero(np.isnan(value_2d))
    return value_2d


def _median_test(value_2d, eps, thresh):
    check = np.zeros((value_2d.shape[0] + 2, value_2d.shape[1] + 2))
    check[:, :] = np.nan
    check[1: value_2d.shape[0] + 1, 1: value_2d.shape[1] + 1] = value_2d
    value_return = np.zeros(value_2d.shape)

    for j in range(1, value_2d.shape[0] + 1):
        for i in range(1, value_2d.shape[1] + 1):
            value_list = np.array([check[j - 1, i - 1], check[j - 1, i], check[j - 1, i + 1], check[j, i - 1],
                                   check[j, i + 1], check[j + 1, i - 1], check[j + 1, i], check[j + 1, i + 1]])
            value_list = value_list[~np.isnan(value_list)]
            value_median = np.median(value_list)
            value_rm = np.median(np.abs(value_list - value_median))

            if np.abs(value_2d[j - 1, i - 1] - value_median) / (value_rm + eps) > thresh:
                value_return[j - 1, i - 1] = np.abs(value_2d[j - 1, i - 1] - value_median) / (value_rm + eps)
    return value_return


def _interpolation(value):
    value_2d = copy.deepcopy(value)
    W = 1/(2**0.5)
    for j in range(1, value_2d.shape[0] - 1):
        for i in range(1, value_2d.shape[1] - 1):
            if np.isnan(value_2d[j, i]):
                value_2d[j, i] = (value_2d[j+1, i]+value_2d[j-1, i]+value_2d[j, i+1]+value_2d[j, i-1])/(4+4*W)+\
                                 (value_2d[j+1, i+1]+value_2d[j+1, i-1]+value_2d[j-1, i+1]+value_2d[j-1, i-1])*W/(4+4*W)

    for j in range(1, value_2d.shape[0] - 1):
        i = 0
        if np.isnan(value_2d[j, i]):
            value_2d[j, i] = (value_2d[j + 1, i] + value_2d[j - 1, i] + value_2d[j, i + 1]) / (3 + 3 * W) + (
                    value_2d[j + 1, i + 1] + value_2d[j - 1, i + 1]) * W / (2 + 2 * W)
        i = value_2d.shape[1] - 1
        if np.isnan(value_2d[j, i]):
            value_2d[j, i] = (value_2d[j + 1, i] + value_2d[j - 1, i] + value_2d[j, i - 1]) / (3 + 3 * W) + (
                    value_2d[j + 1, i - 1] + value_2d[j - 1, i - 1]) * W / (2 + 2 * W)

    for i in range(1, value_2d.shape[1] - 1):
        j = 0
        if np.isnan(value_2d[j, i]):
            value_2d[j, i] = (value_2d[j + 1, i] + value_2d[j, i - 1] + value_2d[j, i + 1]) / (3 + 3 * W) + (
                    value_2d[j + 1, i + 1] + value_2d[j + 1, i - 1]) * W / (2 + 2 * W)
        j = value_2d.shape[0] - 1
        if np.isnan(value_2d[j, i]):
            value_2d[j, i] = (value_2d[j, i + 1] + value_2d[j - 1, i] + value_2d[j, i - 1]) / (3 + 3 * W) + (
                    value_2d[j - 1, i + 1] + value_2d[j - 1, i - 1]) * W / (2 + 2 * W)

    if np.isnan(value_2d[0, 0]):
        value_2d[0, 0] = (value_2d[1, 0] + value_2d[0, 1]) / (2 + 2 * W) + value_2d[1, 1] * W / (1 + 1 * W)
    if np.isnan(value_2d[-1, 0]):
        value_2d[-1, 0] = (value_2d[-2, 0] + value_2d[-1, 1]) / (2 + 2 * W) + value_2d[-2, 1] * W / (1 + 1 * W)
    if np.isnan(value_2d[0, -1]):
        value_2d[0, -1] = (value_2d[0, -2] + value_2d[1, -1]) / (2 + 2 * W) + value_2d[1, -2] * W / (1 + 1 * W)
    if np.isnan(value_2d[-1, -1]):
        value_2d[-1, -1] = (value_2d[-1, -2] + value_2d[-2, -1]) / (2 + 2 * W) + value_2d[-2, -2] * W / (1 + 1 * W)

    return value_2d


def error_vector_interp_2d(vector_x, vector_y, eps=0.3, thresh=5):
    vector_x = _interpolate_nan(vector_x)
    vector_y = _interpolate_nan(vector_y)
    filter_x = _median_test(vector_x, eps=eps, thresh=thresh)
    filter_y = _median_test(vector_y, eps=eps, thresh=thresh)
    filter_error = filter_x + filter_y
    vector_x = _interpolation(vector_x)
    vector_y = _interpolation(vector_y)
    return vector_x, vector_y, filter_error


def error_vector_interp_3d(vector_x, vector_y, vector_z, eps=0.3, thresh=5):
    vector_x = _interpolate_nan(vector_x.copy())
    vector_y = _interpolate_nan(vector_y.copy())
    vector_z = _interpolate_nan(vector_z.copy())
    filter_x = _median_test(vector_x, eps=eps, thresh=thresh)
    filter_y = _median_test(vector_y, eps=eps, thresh=thresh)
    filter_z = _median_test(vector_z, eps=eps, thresh=thresh)
    filter_error = filter_x + filter_y + filter_z
    vector_x = _interpolation(vector_x.copy())
    vector_y = _interpolation(vector_y.copy())
    vector_z = _interpolation(vector_z.copy())
    return vector_x, vector_y, vector_z, filter_error


def smoothing(src, mode='gauss', ksize=21):
    src_save = copy.deepcopy(src)
    src_save[src_save == 0.0] = np.nan
    if mode == 'gauss':
        ret = cv2.GaussianBlur(src, ksize=(ksize, ksize), sigmaX=3)
    elif mode == 'median':
        ret = ndimage.median_filter(src, size=ksize)
    else:
        print('Error')
        return src
    src_expand = np.zeros((src.shape[0] + ksize * 2, src.shape[1] + ksize * 2))
    src_expand[:, :] = np.nan
    src_expand[ksize:-ksize, ksize:-ksize] = src_save
    for j in range(src.shape[0]):
        for i in range(src.shape[1]):
            if np.isnan(src_save[j, i]):
                ret[j, i] = np.nan
            elif np.isnan(ret[j, i]) and ~np.isnan(src_save[j, i]):
                window = src_expand[j - int(ksize // 2) + ksize:j - int(ksize // 2) + 2 * ksize,
                         i - int(ksize // 2) + ksize:i - int(ksize // 2) + 2 * ksize]
                ret[j, i] = np.nanmean(window)
    return ret


def interpolate_sparse_motion(x, y, u, v, domain_size, function="multiquadric", epsilon=None, smooth=0.1, nchunks=10):
    """Interpolation of sparse motion vectors to produce a dense field of motion
    vectors. It uses the scipy.interpolate class Rbf():
    https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.interpolate.Rbf.html
    TODO: use KDTree to speed up interpolation
    Parameters
    ----------
    x : array-like
        x coordinates of the sparse motion vectors
    y : array-like
        y coordinates of the sparse motion vectors
    u : array_like
        u components of the sparse motion vectors
    v : array_like
        v components of the sparse motion vectors
    domain_size : tuple
        size of the domain of the dense motion field [px]
    function : string
        the radial basis function, based on the radius, r, given by the Euclidian
        norm
    epsilon : float
        adjustable constant for gaussian or multiquadrics functions - defaults
        to approximate average distance between nodes (which is a good start)
    smooth : float
        values greater than zero increase the smoothness of the approximation.
        0 is for interpolation, meaning that the function will always go through
        the nodal points in this case
    nchunks : int
        split the grid points in n chunks to limit the memory usage during the
        interpolation

    Returns
    -------
    X : array-like
        grid
    Y : array-like
        grid
    UV : array-like
        Three-dimensional array (2,domain_size[0],domain_size[1])
        containing the dense U, V motion fields.
    """

    # make sure these are vertical arrays
    x = x[:, None]
    y = y[:, None]
    u = u[:, None]
    v = v[:, None]

    points = np.column_stack((x, y))

    if len(domain_size) == 1:
        domain_size = (domain_size, domain_size)

    # generate the grid
    xgrid = np.arange(domain_size[1])
    ygrid = np.arange(domain_size[0])
    X, Y = np.meshgrid(xgrid, ygrid)
    grid = np.column_stack((X.ravel(), Y.ravel()))

    U = np.zeros(grid.shape[0])
    V = np.zeros(grid.shape[0])

    # split grid points in n chunks
    subgrids = np.array_split(grid, nchunks, 0)
    subgrids = [x for x in subgrids if x.size > 0]

    # loop subgrids
    i0 = 0
    for i, subgrid in enumerate(subgrids):
        idelta = subgrid.shape[0]
        # get instances of the radial basis function interpolator
        if i == 0:
            rbfiu = scipy.interpolate.Rbf(x, y, u, function=function, epsilon=epsilon, smooth=smooth)
            rbfiv = scipy.interpolate.Rbf(x, y, v, function=function, epsilon=epsilon, smooth=smooth)

        # interpolate values
        U[i0: (i0 + idelta)] = rbfiu(subgrid[:, 0], subgrid[:, 1])
        V[i0: (i0 + idelta)] = rbfiv(subgrid[:, 0], subgrid[:, 1])

        i0 += idelta

    # reshape back to original size
    U = U.reshape(domain_size[0], domain_size[1])
    V = V.reshape(domain_size[0], domain_size[1])
    UV = np.stack([U, V])

    return X, Y, UV
