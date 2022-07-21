import numpy as np

from rcit.verify import veri_object


def sal_veri(obs, fcst, thresh, vrbl=None, utc=None, lv=None, accum_hr=None, footprint=500, dx=1, dy=1,
             autofactor=1 / 15):
    obs = np.nan_to_num(obs)
    fcst = np.nan_to_num(fcst)
    # shape of obs data and fcst data must be equaled
    n_x, n_y = obs.shape

    obj_obs, nobj_obs, maxobjn_obs, com_obs, R_tot_obs = veri_object.object_identify_veri(obs, thresh, footprint,
                                                                                          autofactor, datamin=None)
    obj_fcst, nobj_fcst, maxobjn_fcst, com_fcst, R_tot_fcst = veri_object.object_identify_veri(fcst, thresh, footprint,
                                                                                               autofactor, datamin=None)

    distance = compute_d(dx, dy, n_x, n_y)
    structure = compute_structure(obj_obs, obj_fcst, R_tot_obs, R_tot_fcst)
    amplitude = compute_amplitude(obs, fcst)
    location = compute_location(dx, obj_obs, obj_fcst, com_obs, com_fcst, R_tot_obs, R_tot_fcst, distance)

    return structure, amplitude, location


def compute_d(dx, dy, n_x, n_y):
    xside = dx * n_x
    yside = dy * n_y
    d = np.sqrt(xside ** 2 + yside ** 2)
    return d


def compute_amplitude(obs, fcst):
    A = (np.mean(fcst) - np.mean(obs)) / (0.5 * (np.mean(fcst) + np.mean(obs)))
    return A


def compute_location(dx, obj_obs, obj_fcst, com_obs, com_fcst, R_tot_obs, R_tot_fcst, d):
    L1 = compute_L1(dx, com_obs, com_fcst, d)
    L2 = compute_L2(dx, obj_obs, obj_fcst, com_obs, com_fcst, R_tot_obs, R_tot_fcst, d)
    L = L1 + L2
    return L


def compute_L1(dx, com_obs, com_fcst, d):
    # vector subtraction
    dist_km = vector_diff_km(dx, com_fcst, com_obs)
    L1 = dist_km / d
    print(("L1 = {0}".format(L1)))
    return L1


def vector_diff_km(dx, v1, v2):
    # From grid coords to km difference
    dist_gp = np.subtract(v1, v2)
    dist_km = dx * np.sqrt(dist_gp[0] ** 2 + dist_gp[1] ** 2)
    return dist_km


def compute_L2(dx, obj_obs, obj_fcst, com_obs, com_fcst, R_tot_obs, R_tot_fcst, d):
    rc = compute_r(dx, obj_obs, com_obs, R_tot_obs)
    rm = compute_r(dx, obj_fcst, com_fcst, R_tot_fcst)
    L2 = 2 * (np.abs(rc - rm) / d)
    print(("L2 = {0}".format(L2)))
    return L2


def compute_r(dx, objects, x_CoM, R_tot):
    Rn_sum = 0
    for k, v in list(objects.items()):
        if np.isnan(v['Rn']) or np.isnan(v['CoM'][0]):
            print("NaN detected in r computationp. Ignoring.")
        else:
            Rn_sum += v['Rn'] * vector_diff_km(dx, x_CoM, v['CoM'])
    try:
        r = Rn_sum / R_tot
    except ZeroDivisionError:
        r = 0
    return r


def compute_structure(obj_obs, obj_fcst, R_tot_obs, R_tot_fcst):
    Vm = compute_V(obj_fcst, R_tot_fcst)
    Vc = compute_V(obj_obs, R_tot_obs)
    try:
        S = (Vm - Vc) / (0.5 * (Vm + Vc))
    except ZeroDivisionError:
        S = 0
    return S


def compute_V(objects, R_tot):
    Vn_sum = 0
    for k, v in list(objects.items()):
        Vn_sum += v['Rn'] * v['Vn']
    try:
        V = Vn_sum / R_tot
    except ZeroDivisionError:
        V = 0
    return V
