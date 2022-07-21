import copy

import numpy as np
import operator as op
cimport numpy as np
import cython
from libc.math cimport sqrt
from libc.stdlib cimport abs

ctypedef np.float64_t float64

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline statistic_parameter_motion(np.ndarray[float64, ndim=1] u):
    cdef float64 std_limit = 2.0
    f = u[~np.isnan(u)]
    f_mean = np.mean(f)
    f_std = np.std(f)

    # removing the values outside the "std_limited" times "std"
    # and recalculate mean and standard dev
    for i in range(len(f)):
        if np.abs(f[i] - f_mean) > std_limit * f_std:
            f[i] = np.nan

    f2 = f[~np.isnan(f)]
    u_mean = np.mean(f2)
    u_std = np.std(f2)
    u_med = np.median(f2)

    return u_mean, u_std, u_med


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def motion_filter_global(np.ndarray[float64, ndim=2] ui,
                         np.ndarray[float64, ndim=2] vi,
                         int r_MM):

    # r_MM typically is 2 or 3
    uo = copy.deepcopy(ui)
    vo = copy.deepcopy(vi)
    # u = [x for x in uo if x.any() != np.nan]
    # v = [x for x in vo if x.any() != np.nan]

    mx = len([x for x in uo if x.any() != np.nan])
    u_mean = np.nanmean(uo)
    v_mean = np.nanmean(vo)
    u_std = np.nanstd(uo)
    v_std = np.nanstd(vo)
    U_std = sqrt(u_std ** 2 + v_std ** 2)

    # for iy in range(my):
    for ix in range(mx):
        r = sqrt((ui[ix]-u_mean)**2+(vi[ix]-v_mean)**2)/U_std
        if r.any() >= r_MM:
            uo[ix] = np.nan
            vo[ix] = np.nan

    return uo, vo


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def motion_filter_local(np.ndarray[float64, ndim=2] motion_vector,
                        float64 threshold,
                        method):
    # window diameter for filtering in the ref_image when an area of (ndia*2+1) by (ndia*2+1) is given
    global f_std
    ndia_start = 1
    ndia_max = 4

    # setting the  minimum number of valid (but unfiltered) neighboring vectors for filtering
    nf_min = 9

    # intrinsic the particle ref_image velocimetry error
    err_std = 0.25

    # beginning the process
    big = 10 ** 5
    i_cond = 0
    vector_u = copy.deepcopy(motion_vector)
    mx, my = vector_u.shape

    if ndia_max < ndia_start:
        raise Exception('ndia_start must < or = ndia_max')
    if mx < ndia_start * 2 + 1 or my < ndia_start * 2 + 1:
        raise Exception('The total number of vectors is too small for filtering')

    for iy in range(my):
        for ix in range(mx):
            iflag = 0
            ndia = ndia_start

            while iflag == 0:
                lx1 = ix - ndia
                lx2 = ix + ndia
                ly1 = iy - ndia
                ly2 = iy + ndia

                if lx1 < 1:
                    lx1 = 0
                if ly1 < 1:
                    ly1 = 0
                if lx2 > mx:
                    lx2 = mx - 1
                if ly2 > my:
                    ly2 = my - 1

                fr = motion_vector[lx1:lx2, ly1:ly2]
                fr = np.reshape(fr, [1, fr.shape[0] * fr.shape[1]])
                f = fr[~np.isnan(fr)]

                if len(f) >= nf_min:
                    mean, std, med = statistic_parameter_motion(f)
                    iflag = 1

                elif ndia == ndia_max:
                    std = 0.0
                    mean = big
                    med = big
                    iflag = 1
                else:
                    ndia = ndia + 1

            default_filter = 'median'
            # if the filter is mean the condition is vector_u_nei[i, 0] - mean
            method = op.eq(method, default_filter)

            if method:
                if abs(motion_vector[ix, iy]-med) > (threshold*std > err_std and threshold*std or err_std):
                    vector_u[ix, iy] = np.nan
            else:
                if abs(motion_vector[ix, iy]-mean) > (threshold*std > err_std and threshold*std or err_std):
                    vector_u[ix, iy] = np.nan

    filtered_motion_vector = vector_u
    return filtered_motion_vector

