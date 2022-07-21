"""
author: Ting He

Summary:

Generic filter interface for different curvature filters.
"""

import numpy as np
from skimage.color.adapt_rgb import adapt_rgb, each_channel

from rcit.util.pre_process.curvature_filter.cf_update import update_bernstein, update_gc, update_mc, update_tv

updaterule = {'tv': update_tv, 'gc': update_gc, 'mc': update_mc, 'bernstein': update_bernstein}


@adapt_rgb(each_channel)
def cf_filter(inputimg, filtertype, total_iter=3, dtype=np.float):
    """
    This function applies curvature filter on input ref_image for 10 iterations (default).

    Parameters:
        inputimg: 2D numpy array that contains ref_image data.
        filtertype: string which is used to define filter type,
                    'mc' for Mean Curvature filter,
                    'gc' for Gaussian Curvature filter,
                    'tv' for Total Variation filter,
                    'bernstein' for Bernstein filter.
        total_iter: number of iterations, default is 10.
        dtype     : numpy datatype to be used for calculation, default datatype
                    is numpy.float32.

    Return:
        2D numpy array, the input ref_image is not modified.
    """
    assert (type(filtertype) is str), "input argument is not a string datatype!"
    assert (filtertype in updaterule.keys()), "filter type is not found!"
    filteredimg = np.copy(inputimg.astype(dtype))
    update = updaterule.get(filtertype)

    for iter_num in range(total_iter):
        update(filteredimg, 1, 1)
        update(filteredimg, 2, 2)
        update(filteredimg, 1, 2)
        update(filteredimg, 2, 1)

    return filteredimg
