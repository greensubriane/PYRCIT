import numpy as np
from numpy import floor, ceil, sqrt, log


def peak_pixel_searching(f,
                         i_opt):
    size_f = f.shape[0] * f.shape[1]
    n0 = np.count_nonzero(~np.isnan(f))
    f_max = np.max(f)

    # remove unusual value
    if n0 == 0 or size_f == 0 or f_max == 0:
        peak_loc_x = np.nan
        peak_loc_y = np.nan
        SNR = np.nan
        MMR = np.nan
        PPR = np.nan
        return

    # peak value searching with normal way
    # searching the first peak
    peak_loc_x = -1
    peak_loc_y = -1
    f_max = f[0, 0]

    for iy in range(f.shape[1]):
        g = f[:, iy]
        g_max = np.max(g)
        ig = np.argmax(g)

        if g_max >= f_max:
            f_max = g_max
            peak_loc_x = ig
        # f_max = f[0, 0]

        for ix in range(f.shape[0]):
            g = f[ix]
            g_max = np.max(g)
            ig = np.argmax(g)

            if f_max <= g_max:
               f_max = g_max
               peak_loc_y = ig

        # searching the second peak
        h = f[peak_loc_x, peak_loc_y]

        ip_x0 = peak_loc_x
        ip_y0 = peak_loc_y

        # peak searching with Gaussian subpixel way:: the gaussian function is given:
        # f(x) = A*exp(-(x0-x)^2/B), it is based on the three point curve fitting
        # the peak value location at horizontal and vertical direction is given seperatly:
        # x0 = 0.5*(lnR(i-1,k)-lnR(i+1,j))/(lnR(i-1,j)-2lnR(i,j)+lnR(i+1,j))
        # y0 = 0.5*(lnR(i,j-1)-lnR(i,j+1))/(lnR(i,j-1)-2lnR(i,j)+2lnR(i,j+1))
        if np.abs(i_opt) == 2:
            if peak_loc_x == -1 or peak_loc_y == -1:
               peak_loc_x = np.nan
               peak_loc_y = np.nan
               return

            # calculating sub-pixel peak value at horizontal position
            g = f[:, peak_loc_y]
            h = np.max(g)
            ih = np.argmax(g)
            g = g/h

            if ih == 0:
               ix_subpeak = 0
            elif ih == f.shape[0] - 1:
               ix_subpeak = f.shape[0] - 1
            else:
               if g[ih + 1] != 0 and g[ih] != 0 and g[ih - 1] != 0:
                   ix_subpeak = ih-0.5*(log(g[ih+1])-log(g[ih-1]))/(log(g[ih+1])-2*log(g[ih])+log(g[ih-1]))
               else:
                   ix_subpeak = np.nan

            # calculating sub-pixel peak value at vertical position
            g = f[peak_loc_x, :]
            h = np.max(g)
            ih = np.argmax(g)
            g = g/h

            if ih == 0:
                iy_subpeak = 0
            elif ih == f.shape[1] - 1:
                iy_subpeak = f.shape[1] - 1
            else:
                if g[ih + 1] != 0 and g[ih] != 0 and g[ih - 1] != 0:
                    iy_subpeak = ih-0.5*(log(g[ih+1])-log(g[ih-1]))/(log(g[ih+1])-2*log(g[ih])+log(g[ih-1]))
                else:
                    iy_subpeak = np.nan

            ip_x0 = peak_loc_x
            ip_y0 = peak_loc_y
            peak_loc_x = ix_subpeak
            peak_loc_y = iy_subpeak

        # searching the second peak calculating the maximum value of the first peak
        peak_1 = h
        g = f
        dia_peak = round(sqrt(3 * f.shape[0]) / 2.5)
        lx1 = ip_x0 - dia_peak
        lx2 = ip_x0 + dia_peak
        ly1 = ip_y0 - dia_peak
        ly2 = ip_y0 + dia_peak

        if lx1 < 1:
             lx1 = 1
        elif lx2 > f.shape[0]:
             lx2 = f.shape[0]

        if ly1 < 1:
             ly1 = 1
        elif ly2 > f.shape[1]:
             ly2 = f.shape[1]
        g[int(lx1):int(lx2), int(ly1):int(ly2)] = 0
        peak_2 = np.max(g)

        # postprocessing and calculating the ratio of signal to noise
        if i_opt > 1:
            g = f
        else:
            g = f[f>0]

        if np.max(g.shape) > 2:
            f_max = g.max()
            f_std = np.std(g)
            f_mean = np.mean(g)
            f_amean = np.mean(np.abs(g))
        else:
            f_max = np.nan
            f_std = np.nan
            f_mean = np.nan
            f_amean = np.nan

        MMR = f_max / f_amean
        SNR = (f_max - f_mean) / f_std

        # calculating the ratio of peak to peak
        if peak_2 != 0:
            PPR = peak_1 / peak_2
        else:
            PPR = np.inf
        return peak_loc_x, peak_loc_y, SNR, MMR, PPR


def position_particle_image_velocimetry(nx,
                                        ny,
                                        nx_pixel,
                                        ny_pixel,
                                        overlap_x,
                                        overlap_y):
    # distance from the ref_image edge which is divided by window size
    n_r = 0  # 1/4, 0
    if nx_pixel%2 == 0:
        center_x = nx_pixel/2+0.5
    else:
        center_x = nx_pixel/2
    if ny_pixel%2 == 0:
        center_y = ny_pixel/2+0.5
    else:
        center_y = ny_pixel/2

    # getting overlapping of sub-window in pixel
    overlaps_x = nx_pixel * overlap_x
    overlaps_y = ny_pixel * overlap_y

    # starting piv process at pixel point
    # defaultpivtype = 'mqd'
    # logicalpivtype = op.eq(piv_type, defaultpivtype)
    start_position_x = nx_pixel * n_r
    start_position_y = ny_pixel * n_r
    # else:
    #     start_position_x = 1
    #     start_position_y = 1

    # getting the number of vectors
    number_horizontal = ceil(((nx-1*start_position_x)-(nx_pixel-overlaps_x))/(nx_pixel-overlaps_x))+1
    number_vertical = ceil(((ny-1*start_position_y)-(ny_pixel-overlaps_y))/(ny_pixel-overlaps_y))+1

    # setting the location of motion vector
    loc_x = []
    loc_y = []
    loc_x_append = loc_x.append
    loc_y_append = loc_y.append

    for ix in range(number_horizontal):
        # origin of horizontal location of target window
        ix1 = start_position_x + (nx_pixel - overlaps_x) * ix + 1
        # getting center location of target window
        # loc_x[ix] = (ix1-1 + center_x)
        loc_x_append(ix1 - 1 + center_x)

    for iy in range(number_vertical):
        # origin of vertical location of target window
        iy1 = start_position_y + (ny_pixel - overlaps_y) * iy + 1
        # getting center location of target window
        # loc_y[iy] = (iy1-1 + center_y)
        loc_y_append(iy1 - 1 + center_y)
    # checking the positions of motion vectors

    while loc_x[number_horizontal - 1] > nx - 1 + 0.5:
        loc_x = loc_x[0:number_horizontal - 1]
        number_horizontal = number_horizontal - 1
    while loc_y[number_vertical - 1] > ny - 1 + 0.5:
        loc_y = loc_y[0:number_vertical - 1]
        number_vertical = number_vertical - 1

    return loc_x, loc_y, center_x, center_y


def minimum_quadric_difference(im1, im2, nx_pixel, ny_pixel, overlap_x, overlap_y, iu_max, iv_max):

    # defining ratio of maximum and mean
    r_MMR = 1.1

    # defining ratio of the first and the second mqd peak value
    r_PPR = 1.05

    # set the percentage of sub window for defining the searching area
    percentage_search = 1/3

    # setting lower and upper value limits for MQD result
    mqd_min = pow(10, -5)
    mqd_max = np.inf

    # getting size of input radar reflectivity ref_image
    # size_row_t1, size_column_t1 = im1.shape
    # size_row_t2, size_column_t2 = im2.shape

    # defining the searching area
    if iu_max <= 0 or iv_max <= 0:
        # searching area is defined by the 1/3 rule
        max_move_distance_horizontal = ceil(percentage_search * nx_pixel)
        max_move_distance_vertical = ceil(percentage_search * ny_pixel)
    else:
        # the searching area is defined by the maximum vector at hori and vect direction
        max_move_distance_horizontal = floor(iu_max)
        max_move_distance_vertical = floor(iv_max)
        if max_move_distance_horizontal >= nx_pixel:
            max_move_distance_horizontal = nx_pixel
        if max_move_distance_vertical >= ny_pixel:
            max_move_distance_vertical = ny_pixel

    # calculating the searching area
    nx_search = int(2*max_move_distance_horizontal+1)
    ny_search = int(2*max_move_distance_vertical+1)

    # obtain the center locations for each sub window
    loc_x, loc_y, dx_center, dy_center = position_particle_image_velocimetry(im1.shape[0],
                                                                             im1.shape[1],
                                                                             nx_pixel,
                                                                             ny_pixel,
                                                                             overlap_x,
                                                                             overlap_y)

    # getting number of vector for each sub window
    is_x = np.zeros((len(loc_x), len(loc_y)), dtype='float')
    is_y = np.zeros((len(loc_x), len(loc_y)), dtype='float')

    # main process
    for i_y in range(len(loc_y)):
        for i_x in range(len(loc_x)):
            # creating the target window from the first ref_image and getting area of target subwindow
            ix1 = loc_x[i_x] - dx_center
            ix2 = ix1 + nx_pixel - 1
            iy1 = loc_y[i_y] - dy_center
            iy2 = iy1 + ny_pixel - 1
            # getting the center of target window
            ix_center = ix1 - 1 + dx_center
            iy_center = iy1 - 1 + dy_center
            f1 = im1[int(ix1):int(ix2), int(iy1):int(iy2)]
            # calculating the MQD value
            c = np.zeros((nx_search, ny_search), dtype=float)
            isy = -1
            move_distance_vertical = np.arange(-max_move_distance_vertical, max_move_distance_vertical + 1)
            move_distance_horizontal = np.arange(-max_move_distance_horizontal, max_move_distance_horizontal + 1)

            for jy in move_distance_vertical:
                isx = -1
                isy = isy + 1
                for jx in move_distance_horizontal:
                    isx = isx + 1
                    # creating the searching sub window from the second window
                    # f2 = np.zeros((nx_pixel, ny_pixel), dtype=float)
                    kx1 = ix1 + jx
                    kx2 = kx1 + nx_pixel - 1
                    ky1 = iy1 + jy
                    ky2 = ky1 + ny_pixel - 1
                    if (kx1 >= 0) and (kx2 <= im1.shape[0] - 1) and (ky1 >= 0) and (ky2 <= im1.shape[1] - 1):
                        f2 = im2[int(kx1):int(kx2), int(ky1):int(ky2)]
                        n = im2.shape[0] * im2.shape[1]
                        d = np.sum(np.sum(np.abs(f1-f2)))/n
                        if d != 0:
                            c[isx, isy] = d
                        else:
                            c[isx, isy] = mqd_min
                    else:
                        c[isx, isy] = mqd_max

            # for simple calculation, the MQD result is diveded by 1
            c = 1. / c
            nx, ny = c.shape

            # calculating the position of the peak value
            peak_loc_x, peak_loc_y, SNR, MMR, PPR = peak_pixel_searching(c, 2)
            if nx % 2 == 0:
                ix_peak = peak_loc_x - (nx / 2 + 0.5)
            else:
                ix_peak = peak_loc_x - ceil(nx / 2)

            if ny % 2 == 0:
                iy_peak = peak_loc_y - (ny / 2 + 0.5)
            else:
                iy_peak = peak_loc_y - ceil(ny / 2)
            if MMR < r_MMR or PPR < r_PPR:
                ix_peak = np.nan
                iy_peak = np.nan

            # eliminating the motion vectors which exceed the upper limits
            if np.abs(ix_peak) >= iu_max:
                ix_peak = np.nan
                iy_peak = np.nan

            if np.abs(iy_peak) >= iv_max:
                ix_peak = np.nan
                iy_peak = np.nan

            is_x[i_x, i_y] = ix_peak
            is_y[i_x, i_y] = iy_peak

    v_x = is_x
    v_y = is_y

    return loc_x, loc_y, v_x, v_y


def particle_image_velocimetry(im1,
                               im2,
                               nx_pixel,
                               ny_pixel,
                               overlap_x,
                               overlap_y,
                               iu_max,
                               iv_max,):

    # preprocessing: transpose of the f ref_image(y,x) to ref_image(x,y)
    im_t1 = np.transpose(im1)
    im_t2 = np.transpose(im2)

    # calculating horizontal and vertical size of ref_image
    # size_row_1, size_column_1 = im1.shape
    # size_row_2, size_column_2 = im2.shape

    if im_t1.shape[0] != im_t2.shape[0] or im_t1.shape[1] != im_t2.shape[1]:
        raise Exception('Error: two radar image sizes are different, exit!')

    nx_pixel_1 = np.around(nx_pixel)
    ny_pixel_1 = np.around(ny_pixel)

    # setting limits of overlap between sub windows, the upper limit is 0.9
    if overlap_x > 0.9 or overlap_y > 0.9:
        raise Exception('Error: the overlap ratio is too large, exit!')

    # selecting the method for calculting the motion vectors,
    # the common way is maximum cross-correlation method,
    # in this study, the minimum quadric difference method is applied (defaultpivtype is 'mqd')
    #defaultpivtype = 'mqd'
    # piv_type = op.eq(piv_type, defaultpivtype)

    # if piv_type:
    # calling the mqd method
    loc_x, loc_y, v_x, v_y = minimum_quadric_difference(im_t1, im_t2,
                                                        nx_pixel_1, ny_pixel_1,
                                                        overlap_x, overlap_y,
                                                        iu_max, iv_max)

        # deriving the motion vector
        # v_x = v_x/dt
        # v_y = v_y/dt

    return loc_x, loc_y, v_x, v_y
