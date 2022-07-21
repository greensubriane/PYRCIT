import numpy as np
cimport numpy as np
import cython

ctypedef np.float64_t float64


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def motion_interp_linear(np.ndarray[float64, ndim=2] motion):
    cdef np.ndarray[float64, ndim=2] uo = motion
    cdef long mx = motion.shape[0]
    cdef long my = motion.shape[1]

    # main process
    cdef np.ndarray[long, ndim=1] mylist = np.arange(1, my-2)
    cdef np.ndarray[long, ndim=1] mxlist = np.arange(1, mx-2)

    for iy in mylist:
        for ix in mxlist:
            if np.isnan(motion[ix, iy]):
                uo[ix, iy] = (motion[ix+1, iy]+motion[ix-1, iy]+motion[ix, iy+1]+motion[ix, iy-1])/4

    # top side
    for ix in mxlist:
        if np.isnan(motion[ix, 0]):
            uo[ix, 0] = motion[ix, 1]
    # bottom side
    for ix in mxlist:
        if np.isnan(motion[ix, my-1]):
            uo[ix, my-1] = motion[ix, my-2]

    # left side
    for iy in mylist:
        if np.isnan(motion[0, iy]):
            uo[0, iy] = motion[1, iy+1]
    # right side
    for iy in mylist:
        if np.isnan(motion[mx-1, iy]):
            uo[mx-1, iy] = motion[mx-2, iy]
    # left-top corner
    if np.isnan(motion[0, 0]):
        uo[0, 0] = motion[1, 1]
    # right-top corner
    if np.isnan(motion[mx-1, 0]):
        uo[mx-1, 0] = motion[mx-2, 1]
    # left-bottom corner
    if np.isnan(motion[0, my-1]):
        uo[0, my-1] = motion[1, my-2]
    # right-bottom corner
    if np.isnan(motion[mx-1, my-1]):
        uo[mx-1, my-1] = motion[mx-2, my-2]
    return uo


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def motion_interp_spline(np.ndarray[float64, ndim=2] motion):
    cdef long mx = motion.shape[0]
    cdef long my = motion.shape[1]
    cdef np.ndarray[float64, ndim=2] uo = motion_interp_linear(motion)
    # main process
    cdef np.ndarray[long, ndim=1] mylist = np.arange(2, my - 3)
    cdef np.ndarray[long, ndim=1] mxlist = np.arange(2, mx - 3)

    for iy in mylist:
        for ix in mxlist:
            if np.isnan(motion[ix, iy]):
                uo[ix, iy] = 1/3*(motion[ix-1, iy]+motion[ix+1, iy]+motion[ix, iy-1]+motion[ix, iy+1])-\
                             1/12*(motion[ix-2, iy]+motion[ix+2, iy]+motion[ix, iy-2]+motion[ix, iy+2])
    return uo

'''
def distance_matrix(x0, y0, x1, y1):
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
    d1 = np.subtract.outer(obs[:, 1], interp[:, 1])

    return np.hypot(d0, d1)
'''

'''
def idw_interpolation(x, y, z, xi, yi):
    """
    Inverse weighted distance based interpolation algorithm
    :param x: vectors of coordinates for the x axis
    :param y: vectors of coordinates for the y axis
    :param z: vector of dutycycle values
    :param xi: grid of location points used for interpolation
    :param yi: grid of location points used for interpolation
    :return: zi (vector of calculated/interpolated values)
    """

    dist = distance_matrix(x, y, xi, yi)

    # In IDW, weights are 1 / distance --> this can change based on some propagation model preferences and information
    weights = 1.0 / dist

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi
'''