import copy
import math as mt
from pydacefit.corr import corr_gauss
from pydacefit.dace import DACE, regr_quadratic

import numpy as np
# import cupy as cp


def motion_interp_kirgin(motion):
    # gp = GaussianProcessRegressor(theta0 = 0.1, thetaL = .001, thetaU = 1., nugget = 0.01)
    # theta = 1.0
    # regression can be: regr_constant, regr_linear or regr_quadratic
    # regression = regr_constant
    # regression = regr_linear
    regression = regr_quadratic
    # then define the correlation (all possible correlations are shown below)
    # please have a look at the MATLAB document for more details
    correlation = corr_gauss
    # correlation = corr_cubic
    # correlation = corr_exp
    # correlation = corr_expg
    # correlation = corr_spline
    # correlation = corr_spherical
    # correlation = corr_cubic
    mx = motion.shape[0]
    my = motion.shape[1]
    nv = motion.shape[0] * motion.shape[1]
    S1 = []
    S2 = []
    Y = []
    n = -1
    for ix in range(mx):
        for iy in range(my):
            if np.isnan(motion[ix, iy]) == False:
                n = n + 1
                S1.append(ix)
                S2.append(iy)
                Y.append(motion[ix, iy])
    S1 = np.asarray(S1)
    S2 = np.asarray(S2)
    Y = np.asarray(Y)
    S = np.stack((S1, S2))
    X1 = []
    append_X1 = X1.append
    X2 = []
    append_X2 = X2.append

    if n >= 1:
        # dacefit_no_hyperparameter_optimization = DACE(regr=regression, corr=correlation, theta=theta,
        #                                               thetaL=None, thetaU=None)
        # dacefit_with_ard = DACE(regr=regression, corr=correlation, theta=[1.0, 1.0], thetaL=[0.001, 0.0001],
        #                         thetaU=[20, 20])

        dacefit = DACE(regr=regression, corr=correlation, theta=1.0, thetaL=0.00001, thetaU=10)
        dacefit.fit(np.transpose(S), Y)
        # gp.fit(X = np.column_stack([rr[vals], cc[vals]]), y = motion[vals])
        n = -1
        for ix in range(mx):
            for iy in range(my):
                n = n + 1
                # X1.append(ix)
                append_X1(ix)
                # X2.append(iy)
                append_X2(iy)

        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        X = np.stack((X1, X2))
        YX = dacefit.predict(np.transpose(X))
        uo = np.reshape(YX, (my, mx))
    else:
        uo = motion
    uo = np.transpose(uo)
    return uo


def count_nan(fi):
    # f1 = fi
    tmp = np.where(~np.isnan(fi))
    f2 = fi[tmp]
    m = fi.shape[0] * fi.shape[1]
    n = f2.shape[0] * f2.shape[1]
    return m, n

# interpolate NaN vector by kriging interpolation
# Local filtering
# Correlation function: Exponential
# Regression model: First order polynomial


def motion_interp_kriging_local(motion):
    # window size for krigin filter. This must be odd number
    nw_start = 5
    nw_max = 9
    # number of minimum point for interpolation
    np_min = 15
    if np.max(np.max(np.where(np.isnan(motion)))) == 1:
        # initial setup
        # mx, my = motion.shape
        # mv = mx * my

        mx = motion.shape[0]
        my = motion.shape[1]
        nv = motion.shape[0] * motion.shape[1]

        nwh_start = mt.floor(nw_start / 2)
        # main part
        uo = copy.deepcopy(motion)
        if (mx >= nw_start) and (my >= nw_start):
            for ix in range(mx):
                for iy in range(my):
                    if np.isnan(motion[ix, iy]):
                        i_flag = 0
                        nw = nw_start
                        nw_h = nwh_start
                        while not i_flag:
                            # making target area to interpolate
                            lx1 = ix - nw_h
                            lx2 = ix + nw_h
                            ly1 = iy - nw_h
                            ly2 = iy + nw_h
                            jx = nw_h + 1
                            jy = nw_h + 1

                            if lx1 < 1:
                                jx = jx - (1 - lx1)
                                lx2 = lx2 + (1 - lx1)
                                lx1 = 1
                            if lx2 > mx:
                                jx = jx + (lx2 - mx)
                                lx1 = lx1 - (lx2 - mx)
                                lx2 = mx

                            if ly1 < 1:
                                jy = jy - (1 - ly1)
                                ly2 = ly2 + (1 - ly1)
                                ly1 = 1
                            if ly2 > my:
                                jy = jy + (ly2 - my)
                                ly1 = ly1 - (ly2 - my)
                                ly2 = my

                            if (lx1 >= 1) and (ly1 >= 1):
                                fi = motion[lx1:lx2, ly1:ly2]
                            else:
                                fi = np.nan
                            m, n = count_nan(fi)
                            if n < np_min:
                                nw = nw + 2
                                nw_h = nw_h + 1
                            else:
                                i_flag = 1
                            if nw > nw_max:
                                i_flag = 9
                        # kriging interpolation
                        if i_flag == 1:
                            fo = motion_interp_kirgin(fi)
                            uo[ix, iy] = fo[jx, jy]
                        else:
                            uo[ix, iy] = np.nan
        else:
            uo = motion_interp_kirgin(motion)
    else:
        uo = motion
    return uo

