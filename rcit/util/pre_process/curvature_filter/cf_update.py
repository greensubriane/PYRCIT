"""
author: Ting He

Summary:

Update schemes for Total Variation, Mean Curvature, Gaussian Curvature and Bernstein filters.
Please refer to https://github.com/YuanhaoGong/CurvatureFilter for mathematical details behind
these filters. Regular user should not use or modify them, use cf_filter instead.
"""

from __future__ import division

import numpy as np


def update_tv(inputdata, rowbegin, colbegin):
    """
    Update scheme for Total Variation filter.
    Helper function, regular user should not use or modify it.
    """
    inputdata_ = 5 * np.copy(inputdata[rowbegin:-1:2, colbegin:-1:2])
    inputdata_ij = inputdata[rowbegin:-1:2, colbegin:-1:2]

    d1 = (inputdata[rowbegin - 1:-2:2, colbegin:-1:2] + inputdata[rowbegin + 1::2, colbegin:-1:2]) \
         + inputdata[rowbegin:-1:2, colbegin + 1::2] \
         + inputdata[rowbegin - 1:-2:2, colbegin + 1::2] \
         + inputdata[rowbegin + 1::2, colbegin + 1::2] \
         - inputdata_

    d2 = (inputdata[rowbegin:-1:2, colbegin - 1:-2:2] + inputdata[rowbegin:-1:2, colbegin + 1::2]) \
         + inputdata[rowbegin - 1:-2:2, colbegin:-1:2] \
         + inputdata[rowbegin - 1:-2:2, colbegin - 1:-2:2] \
         + inputdata[rowbegin - 1:-2:2, colbegin + 1::2] \
         - inputdata_

    d3 = (inputdata[rowbegin - 1:-2:2, colbegin:-1:2] + inputdata[rowbegin + 1::2, colbegin:-1:2]) \
         + inputdata[rowbegin:-1:2, colbegin - 1:-2:2] \
         + inputdata[rowbegin - 1:-2:2, colbegin - 1:-2:2] \
         + inputdata[rowbegin + 1::2, colbegin - 1:-2:2] \
         - inputdata_

    d4 = (inputdata[rowbegin:-1:2, colbegin - 1:-2:2] + inputdata[rowbegin:-1:2, colbegin + 1::2]) \
         + inputdata[rowbegin + 1::2, colbegin:-1:2] \
         + inputdata[rowbegin + 1::2, colbegin - 1:-2:2] \
         + inputdata[rowbegin + 1::2, colbegin + 1::2] \
         - inputdata_

    d5 = inputdata[rowbegin - 1:-2:2, colbegin + 1::2] + inputdata[rowbegin:-1:2, colbegin + 1::2] \
         + inputdata[rowbegin + 1::2, colbegin:-1:2] + inputdata[rowbegin + 1::2, colbegin + 1::2] \
         + inputdata[rowbegin + 1::2, colbegin - 1:-2:2] \
         - inputdata_

    d6 = inputdata[rowbegin - 1:-2:2, colbegin + 1::2] + inputdata[rowbegin:-1:2, colbegin + 1::2] \
         + inputdata[rowbegin - 1:-2:2, colbegin:-1:2] + inputdata[rowbegin + 1::2, colbegin + 1::2] \
         + inputdata[rowbegin - 1:-2:2, colbegin - 1:-2:2] \
         - inputdata_

    d7 = inputdata[rowbegin - 1:-2:2, colbegin + 1::2] + inputdata[rowbegin:-1:2, colbegin - 1:-2:2] \
         + inputdata[rowbegin - 1:-2:2, colbegin:-1:2] + inputdata[rowbegin:-1:2, colbegin - 1:-2:2] \
         + inputdata[rowbegin + 1::2, colbegin - 1:-2:2] \
         - inputdata_

    d8 = inputdata[rowbegin + 1::2, colbegin + 1::2] + inputdata[rowbegin + 1::2, colbegin - 1:-2:2] \
         + inputdata[rowbegin + 1::2, colbegin:-1:2] + inputdata[rowbegin:-1:2, colbegin - 1:-2:2] \
         + inputdata[rowbegin - 1:-2:2, colbegin - 1:-2:2] \
         - inputdata_

    d = d1 * (np.abs(d1) <= np.abs(d2)) + d2 * (np.abs(d2) < np.abs(d1))
    d = d * (np.abs(d) <= np.abs(d3)) + d3 * (np.abs(d3) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d4)) + d4 * (np.abs(d4) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d4)) + d5 * (np.abs(d4) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d4)) + d6 * (np.abs(d4) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d4)) + d7 * (np.abs(d4) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d4)) + d8 * (np.abs(d4) < np.abs(d))

    d /= 5

    inputdata_ij[...] += d


# @nb.jit()
def update_mc(inputdata, rowbegin, colbegin):
    """
    Update scheme for Mean Curvature filter.
    Helper function, regular user should not use or modify it.
    """
    inputdata_ = 8 * np.copy(inputdata[rowbegin:-1:2, colbegin:-1:2])
    inputdata_ij = inputdata[rowbegin:-1:2, colbegin:-1:2]

    d1 = 2.5 * (inputdata[rowbegin - 1:-2:2, colbegin:-1:2] +
                inputdata[rowbegin + 1::2, colbegin:-1:2]) + \
         5.0 * inputdata[rowbegin:-1:2, colbegin + 1::2] - \
         inputdata[rowbegin - 1:-2:2, colbegin + 1::2] - \
         inputdata[rowbegin + 1::2, colbegin + 1::2] - inputdata_

    d2 = 2.5 * (inputdata[rowbegin:-1:2, colbegin - 1:-2:2] +
                inputdata[rowbegin:-1:2, colbegin + 1::2]) + \
         5.0 * inputdata[rowbegin - 1:-2:2, colbegin:-1:2] - \
         inputdata[rowbegin - 1:-2:2, colbegin - 1:-2:2] - \
         inputdata[rowbegin - 1:-2:2, colbegin + 1::2] - inputdata_

    d3 = 2.5 * (inputdata[rowbegin - 1:-2:2, colbegin:-1:2] +
                inputdata[rowbegin + 1::2, colbegin:-1:2]) + \
         5.0 * inputdata[rowbegin:-1:2, colbegin - 1:-2:2] - \
         inputdata[rowbegin - 1:-2:2, colbegin - 1:-2:2] - \
         inputdata[rowbegin + 1::2, colbegin - 1:-2:2] - inputdata_

    d4 = 2.5 * (inputdata[rowbegin:-1:2, colbegin - 1:-2:2] +
                inputdata[rowbegin:-1:2, colbegin + 1::2]) + \
         5.0 * inputdata[rowbegin + 1::2, colbegin:-1:2] - \
         inputdata[rowbegin + 1::2, colbegin - 1:-2:2] - \
         inputdata[rowbegin + 1::2, colbegin + 1::2] - inputdata_

    d = d1 * (np.abs(d1) <= np.abs(d2)) + d2 * (np.abs(d2) < np.abs(d1))
    d = d * (np.abs(d) <= np.abs(d3)) + d3 * (np.abs(d3) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d4)) + d4 * (np.abs(d4) < np.abs(d))

    d /= 8

    inputdata_ij[...] += d


# @nb.jit()
def update_gc(inputdata, rowbegin, colbegin):
    """
    Update scheme for Gaussian Curvature filter.
    Helper function, regular user should not use or modify it.
    """
    inputdata_ij = inputdata[rowbegin:-1:2, colbegin:-1:2]

    d1 = (inputdata[rowbegin - 1:-2:2, colbegin:-1:2] +
          inputdata[rowbegin + 1::2, colbegin:-1:2]) / 2.0 - inputdata[rowbegin:-1:2, colbegin:-1:2]

    d2 = (inputdata[rowbegin:-1:2, colbegin - 1:-2:2] +
          inputdata[rowbegin:-1:2, colbegin + 1::2]) / 2.0 - inputdata[rowbegin:-1:2, colbegin:-1:2]

    d3 = (inputdata[rowbegin - 1:-2:2, colbegin - 1:-2:2] +
          inputdata[rowbegin + 1::2, colbegin + 1::2]) / 2.0 - inputdata[rowbegin:-1:2, colbegin:-1:2]

    d4 = (inputdata[rowbegin - 1:-2:2, colbegin + 1::2] +
          inputdata[rowbegin + 1::2, colbegin - 1:-2:2]) / 2.0 - inputdata[rowbegin:-1:2, colbegin:-1:2]

    d5 = (inputdata[rowbegin - 1:-2:2, colbegin:-1:2] +
          inputdata[rowbegin:-1:2, colbegin - 1:-2:2] +
          inputdata[rowbegin - 1:-2:2, colbegin - 1:-2:2]) / 3.0 - inputdata[rowbegin:-1:2, colbegin:-1:2]

    d6 = (inputdata[rowbegin - 1:-2:2, colbegin:-1:2] +
          inputdata[rowbegin:-1:2, colbegin + 1::2] +
          inputdata[rowbegin - 1:-2:2, colbegin + 1::2]) / 3.0 - inputdata[rowbegin:-1:2, colbegin:-1:2]

    d7 = (inputdata[rowbegin:-1:2, colbegin - 1:-2:2] +
          inputdata[rowbegin + 1::2, colbegin:-1:2] +
          inputdata[rowbegin + 1::2, colbegin - 1:-2:2]) / 3.0 - inputdata[rowbegin:-1:2, colbegin:-1:2]

    d8 = (inputdata[rowbegin:-1:2, colbegin + 1::2] +
          inputdata[rowbegin + 1::2, colbegin:-1:2] +
          inputdata[rowbegin + 1::2, colbegin + 1::2]) / 3.0 - inputdata[rowbegin:-1:2, colbegin:-1:2]

    d = d1 * (np.abs(d1) <= np.abs(d2)) + d2 * (np.abs(d2) < np.abs(d1))
    d = d * (np.abs(d) <= np.abs(d3)) + d3 * (np.abs(d3) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d4)) + d4 * (np.abs(d4) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d5)) + d5 * (np.abs(d5) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d6)) + d6 * (np.abs(d6) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d7)) + d7 * (np.abs(d7) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d8)) + d8 * (np.abs(d8) < np.abs(d))

    inputdata_ij[...] += d


# @nb.jit()
def update_bernstein(inputdata, rowbegin, colbegin):
    """
    Update scheme for Bernstein filter.
    Helper function, regular user should not use or modify it.
    """
    inputdata_ij = inputdata[rowbegin:-1:2, colbegin:-1:2]

    d1 = (inputdata[rowbegin - 1:-2:2, colbegin:-1:2] + inputdata[rowbegin + 1::2, colbegin:-1:2]) / 2.0 \
         - inputdata[rowbegin:-1:2, colbegin:-1:2]
    d2 = (inputdata[rowbegin:-1:2, colbegin - 1:-2:2] + inputdata[rowbegin:-1:2, colbegin + 1::2]) / 2.0 \
         - inputdata[rowbegin:-1:2, colbegin:-1:2]

    d = d1 * (np.abs(d1) <= np.abs(d2)) + d2 * (np.abs(d2) < np.abs(d1))

    inputdata_ij[...] += d
