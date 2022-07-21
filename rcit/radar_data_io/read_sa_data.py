# -*- coding: utf-8 -*-

"""
read radar data
@author: Ting He
"""

from array import array
from bz2 import BZ2File

import numpy as np
import wradlib as wrl


# read radar binary data
def radar_read(file_path):
    pi = np.pi
    # 读数据
    flag = BZ2File(file_path, "rb")
    data = np.asarray(array("B", flag.read()))
    data = data.reshape([int(len(data) / 2432), 2432])
    # 找仰角
    if data[0, 72] == 11:
        phi = [0.50, 0.50, 1.45, 1.45, 2.40, 3.35, 4.30, 5.25, 6.2, 7.5, 8.7, 10, 12, 14, 16.7, 19.5]
    if data[0, 72] == 21:
        phi = [0.50, 0.50, 1.45, 1.45, 2.40, 3.35, 4.30, 6.00, 9.00, 14.6, 19.5]
    if data[0, 72] == 31:
        phi = [0.50, 0.50, 1.50, 1.50, 2.50, 2.50, 3.50, 4.50]
    if data[0, 72] == 32:
        phi = [0.50, 0.50, 2.50, 3.50, 4.50]
    g1 = np.zeros([len(data), 460])  # 460
    h1 = np.zeros([len(data), 460])  # 460
    i1 = np.zeros([len(data), 460])  # 460
    j1 = np.zeros([len(data), 460])  # 460
    count = 0
    while count < len(data):
        # print("径向数据编号 : ", count)
        b1 = data[count, 44] + 256 * data[count, 45]  # 仰角序数
        c1 = (data[count, 36] + 256 * data[count, 37]) / 8 * 180 / 4096  # 方位角
        d1 = data[count, 54] + 256 * data[count, 55]  # 径向库
        # print("仰角序数,方位角,径向库 : ", b1, c1, d1)
        if d1 == 0:
            count += 1
            continue
        else:
            count += 1
        i = 0
        while i < 460:  # 460
            g1[count - 1, i] = phi[b1 - 1]  # 仰角
            h1[count - 1, i] = c1  # 方位角
            i1[count - 1, i] = 0.5 + i - 1  # 径向
            if i > d1:  # 反射率
                j1[count - 1, i] = 0
            else:
                if data[count - 1, 128 + i] == 0:  # 无数据
                    j1[count - 1, i] = 0
                else:
                    if data[count - 1, 128 + i] == 1:  # 距离模糊
                        j1[count - 1, i] = 0
                    else:  # 数据正常
                        j1[count - 1, i] = (data[count - 1, 128 + i] - 2) / 2 - 32
            i += 1
    n = 1  # 选择仰角，这里共有九层，此处使用第一层
    a2 = 0  # 仰角序数
    while a2 < len(data):
        if data[a2, 44] > (n - 1):
            break
        a2 += 1
    a3 = a2
    while a3 < len(data):
        if data[a3, 44] > n:
            break
        a3 += 1
    yj = g1[a2:a3, :]
    fwj = h1[a2:a3, :]
    jx = i1[a2:a3, :]
    fsl = j1[a2:a3, :]
    return yj, fwj, jx, fsl


def read_china_sa_radar_data(filename, radar_loc_x, radar_loc_y, radar_elev, radar_pol_range):
    # 获取反射率因子
    y, fw, j, fs = radar_read(filename)

    # clutter校正，此处采用Gabella方法
    clutter = wrl.clutter.filter_gabella(fs, wsize=5, thrsnorain=0., tr1=6., n_p=8, tr2=1.3)

    # 生成clutter校正后的反射率因子
    data_no_clutter = wrl.ipol.interpolate_polar(fs, clutter)

    # # 进行回波衰减校正，获取衰减因子, 得到经过衰减校正后的反射率因子
    pia_harrison = wrl.atten.correct_attenuation_hb(data_no_clutter,
                                                    coefficients=dict(a=4.57e-5, b=0.731, gate_length=1.0),
                                                    mode="warn", thrs=75.)
    pia_harrison[pia_harrison > 4.8] = 4.8
    data_attcorr = data_no_clutter + pia_harrison

    # transfer decibel into rainfall intensity
    data_Z = wrl.trafo.idecibel(data_attcorr)
    intensity = wrl.zr.z_to_r(data_Z, a=200., b=1.6)

    radar_location = (radar_loc_x, radar_loc_y, radar_elev)
    # lon, lat, alt = wrl.georef.spherical_to_xyz(j*1000, fw, y, radar_location)

    coords, rad = wrl.georef.spherical_to_xyz(j * 1000, fw, y, radar_location)
    x = coords[..., 0]
    y = coords[..., 1]

    wgs84 = wrl.georef.epsg_to_osr(4326)
    wgs84_coords = wrl.georef.reproject(coords, projection_source=rad, projection_target=wgs84)
    # x, y = wrl.georef.reproject(lon, lat, projection_target=wgs84)
    xgrid = np.linspace(wgs84_coords[..., 0].min(), wgs84_coords[..., 0].max(), 460)  # 920
    ygrid = np.linspace(wgs84_coords[..., 1].min(), wgs84_coords[..., 1].max(), 460)  # 920
    grid_xy = np.meshgrid(xgrid, ygrid)
    grid_xy = np.vstack((grid_xy[0].ravel(), grid_xy[1].ravel())).transpose()
    xy = np.concatenate([wgs84_coords[..., 0].ravel()[:, None], wgs84_coords[..., 1].ravel()[:, None]], axis=1)

    gridded_ref = wrl.comp.togrid(xy, grid_xy, radar_pol_range,
                                  np.array([wgs84_coords[..., 0].mean(), wgs84_coords[..., 1].mean()]),
                                  data_attcorr.ravel(), wrl.ipol.Nearest)  # 460000

    gridded_intensity = wrl.comp.togrid(xy, grid_xy, radar_pol_range,
                                        np.array([wgs84_coords[..., 0].mean(), wgs84_coords[..., 1].mean()]),
                                        intensity.ravel(), wrl.ipol.Nearest)  # 460000

    gridded_refs = np.ma.masked_invalid(gridded_ref).reshape((len(xgrid), len(ygrid)))
    gridded_intensities = np.ma.masked_invalid(gridded_intensity).reshape((len(xgrid), len(ygrid)))

    where_are_inf_ref = np.isinf(gridded_refs)
    gridded_refs[where_are_inf_ref] = 0

    where_are_inf_intensity = np.isinf(gridded_intensities)
    gridded_intensities[where_are_inf_intensity] = 0

    return gridded_refs, gridded_intensities
