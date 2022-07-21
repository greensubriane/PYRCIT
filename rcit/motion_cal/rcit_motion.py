import numpy as np
import skimage.exposure as exp

from rcit.motion_cal.piv_motion import particle_image_velocimetry
from rcit.util.post_process.motion_filter.motion_filter import motion_filter_local

from rcit.util.pre_process.pre_process import isolate_echo_remove
from rcit.util.post_process.post_process import motion_interp_nan
from rcit.util.post_process.post_process import error_vector_interp_2d
from rcit.util.post_process.post_process import smoothing
from rcit.util.post_process.post_process import motion_interp_option
from rcit.util.post_process.post_process import interpolate_sparse_motion


def motion_statistic(time_list, ref_radar, nx_pixel, ny_pixel, overlap_x, overlap_y, iu_max, iv_max, filt_thres):
    v_mat = [[] for i in range(0, time_list)]
    vectors = np.zeros((time_list, 2), dtype=float)

    time_lists = np.arange(0, time_list-1)
    for i in time_lists:
        print(i)
        i = int(i)
        ref_img_t1 = isolate_echo_remove(ref_radar[i], n=3, thr=np.min(ref_radar[i]))
        ref_img_t2 = isolate_echo_remove(ref_radar[i + 1], n=3, thr=np.min(ref_radar[i + 1]))

        loc_x, loc_y, v_x, v_y = particle_image_velocimetry(ref_img_t1, ref_img_t2, nx_pixel,
                                                            ny_pixel,
                                                            overlap_x,
                                                            overlap_y,
                                                            iu_max,
                                                            iv_max)

        v_x_full = motion_interp_nan(v_x)
        v_y_full = motion_interp_nan(v_y)

        v_filt_x = motion_filter_local(v_x_full, filt_thres, 'median')
        v_filt_y = motion_filter_local(v_y_full, filt_thres, 'median')

        v_interp_x = motion_interp_option(v_filt_x, 3)
        v_interp_y = motion_interp_option(v_filt_y, 3)

        loc_grid_x, loc_grid_y = np.meshgrid(loc_x, loc_y)

        grid_x_init, grid_y_init, v_final = interpolate_sparse_motion(loc_grid_x,
                                                                      loc_grid_y,
                                                                      v_interp_x,
                                                                      v_interp_y,
                                                                      (ref_img_t1.shape[0], ref_img_t1.shape[1]),
                                                                      function="multiquadric",
                                                                      epsilon=None,
                                                                      smooth=0.5,
                                                                      nchunks=10)

        v_final_x_1, v_final_y_1, filter_err = error_vector_interp_2d(v_final[0, :, :], v_final[1, :, :], 0.3, 5)

        v_final_x_1 = smoothing(v_final_x_1, mode='gauss', ksize=21)
        v_final_y_1 = smoothing(v_final_y_1, mode='gauss', ksize=21)

        velocity = np.zeros((2, v_final_x_1.shape[0], v_final_x_1.shape[1]), dtype=float)

        velocity[0, :, :] = v_final_x_1
        velocity[1, :, :] = v_final_y_1

        v_mat[i] = velocity

        # generating the prevailing wind direction for each time step according to the generated motion vector
        direction = np.arctan2(v_final_x_1, v_final_y_1)*180/np.pi
        wind_direction = direction+180*np.ones(direction.shape, dtype=float)
        wind_direction_hist, wind_direction_bins = exp.histogram(wind_direction[np.isfinite(wind_direction)], nbins=1)

        # generating the mean speed for each time step according to the generated motion vector
        speeds = np.sqrt(np.power(v_final_x_1, 2)+np.power(v_final_y_1, 2))
        speed_1 = np.reshape(speeds, (speeds.shape[0]*speeds.shape[1], 1))
        wind_speed = speed_1[~np.isnan(speed_1).any(axis=1), :]
        med_speed = np.nanmean(wind_speed)

        vectors[i, 0] = wind_direction_bins
        vectors[i, 1] = med_speed

        loc = [grid_x_init, grid_y_init]

        print('prevailing wind direction is: ', wind_direction_bins, 'mean speed is: ', med_speed)

    return loc, v_mat, vectors

