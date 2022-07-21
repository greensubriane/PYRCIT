import numpy as np
from numpy.matlib import repmat
import copy as copy

""" Generating neighborhoods dataset for each radar pixel.
    Inputs are radar reflectivity data matrix (data) and neighbor size (nei_size).
    For input data matrix, all NaN value should be initialized to 0.
    For input nei_size, there are two options: 4 ane 8, 
    which means 4-neighborhoods and 8-neighborhoods.
"""


def neighbors_four(data, nei_size=4):
    rows, cols = data.shape
    nrc = rows * cols
    In = np.isnan(data).astype(int)

    # replace values in data by index vector
    X = np.reshape(np.arange(1, nrc + 1, dtype=np.float32), (rows, cols), order='A')
    X[In == 1] = np.NaN

    # Pad array ........
    ic = np.full((rows+2, cols+2), np.nan)
    rows_ic, cols_ic = ic.shape
    ic[1:rows_ic-1, 1:cols_ic-1] = X
    I = np.invert(np.isnan(ic)).astype(int)
    icd = np.full((np.count_nonzero(I), nei_size), np.nan)

    # neighbor at right side
    right = np.roll(I[:, :], shift=1, axis=1)
    icd[:, 0] = ic[right.astype(bool)]

    # neighbor at down side
    down = np.roll(I[:, :], shift=1, axis=0)
    icd[:, 1] = ic[down.astype(bool)]

    # neighbor at left side
    left = np.roll(I[:, :], shift=-1, axis=1)
    icd[:, 2] = ic[left.astype(bool)]

    # neighbor at up side
    up = np.roll(I[:, :], shift=-1, axis=0)
    icd[:, 3] = ic[up.astype(bool)]

    # create output
    ic1 = np.transpose(repmat(ic[I.astype(bool)[:]], 1, nei_size))

    neighbor_matrix = np.full((rows*cols, nei_size + 1), np.nan)
    neighbor_matrix[:, 0] = ic1[0:rows * cols, 0]
    neighbor_matrix[:, 1:nei_size + 1] = icd[:, 0:nei_size]
    neighbor_pixels = np.full([rows * cols, nei_size + 1], np.nan)
    # data = np.transpose(data)
    data = np.reshape(data, [rows * cols, 1])
    index = np.argwhere(np.isnan(data))[:, 0]
    data = np.delete(data, index)
    tt = neighbor_matrix[:, 1:nei_size + 1]
    tt = np.nan_to_num(tt)
    neighbor_pixels[:, 0] = data[:]
    index_x = list(range(0, rows * cols))
    index_y = list(range(1, nei_size + 1))
    for t in index_x:
        for k in index_y:
            if int(tt[t, k - 1]) > 0:
                neighbor_pixels[t, k] = data[int(tt[t, k - 1]) - 1]
    return neighbor_pixels


# @nb.jit()
def neighbors_eight(data, nei_size=8):
    rows, cols = data.shape
    nrc = rows * cols
    In = np.isnan(data).astype(int)

    # replace values in data by index vector
    X = np.reshape(np.arange(1, nrc + 1, dtype=np.float32), (rows, cols))
    X[In == 1] = np.NaN

    # Pad array ........
    ic = np.full((rows+2, cols+2), np.nan)
    rows_ic, cols_ic = ic.shape
    ic[1:rows_ic-1, 1:cols_ic-1] = X[:]
    I = np.invert(np.isnan(ic)).astype(int)
    icd = np.full((np.count_nonzero(I), nei_size), np.nan)

    # neighbor at right side
    right = np.roll(I[:, :], shift=1, axis=1)
    icd[:, 0] = ic[right.astype(bool)]

    # neighbor at down side
    down = np.roll(I[:, :], shift=1, axis=0)
    icd[:, 1] = ic[down.astype(bool)]

    # neighbor at left side
    left = np.roll(I[:, :], shift=-1, axis=1)
    icd[:, 2] = ic[left.astype(bool)]

    # neighbor at up side
    up = np.roll(I[:, :], shift=-1, axis=0)
    icd[:, 3] = ic[up.astype(bool)]

    # neighbor at up right side
    up_right = np.roll(right[:, :], shift=-1, axis=0)
    icd[:, 4] = ic[up_right.astype(bool)]

    # neighbor at up left side
    up_left = np.roll(left[:, :], shift=-1, axis=0)
    icd[:, 5] = ic[up_left.astype(bool)]

    # neighbor at down right side
    down_right = np.roll(right[:, :], shift=1, axis=0)
    icd[:, 6] = ic[down_right.astype(bool)]

    # neighbor at down left side
    down_left = np.roll(left[:, :], shift=1, axis=0)
    icd[:, 7] = ic[down_left.astype(bool)]

    # create output
    ic1 = np.transpose(repmat(ic[I.astype(bool)[:]], 1, nei_size))

    neighbor_matrix = np.full((rows*cols, nei_size + 1), np.nan)
    neighbor_matrix[:, 0] = ic1[0:rows * cols, 0]
    neighbor_matrix[:, 1:nei_size + 1] = icd[:, 0:nei_size]
    neighbor_pixels = np.full([rows * cols, nei_size + 1], np.nan)
    # data = np.transpose(data)
    data = np.reshape(data, [rows * cols, 1])
    index = np.argwhere(np.isnan(data))[:, 0]
    data = np.delete(data, index)
    tt = neighbor_matrix[:, 1:nei_size + 1]
    tt = np.nan_to_num(tt)
    neighbor_pixels[:, 0] = data[:]
    index_x = list(range(0, rows * cols))
    index_y = list(range(1, nei_size + 1))
    for t in index_x:
        for k in index_y:
            if int(tt[t, k - 1]) > 0:
                neighbor_pixels[t, k] = data[int(tt[t, k - 1]) - 1]
    return neighbor_pixels

