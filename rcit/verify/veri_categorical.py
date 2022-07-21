import numpy as np


def CSI(a, b, c):
    if a + b + c == 0:
        return None
    return np.round(a / (a + b + c), 3)


def FAR(a, b):
    if a + b == 0:
        return None
    return np.round(b / (a + b), 3)


def POD(a, c):
    if a + c == 0:
        return None
    return np.round(a / (a + c), 3)


def ACC(a, b, c, d):
    if a + b + c + d == 0:
        return None
    return np.round((a + d) / (a + b + c + d), 3)


def categorical_cal(fcst, obs, thr):
    fcst[np.where(fcst == np.nan)] = 0.0
    obs[np.where(obs == np.nan)] = 0.0
    a = 0
    b = 0
    c = 0
    d = 0
    fcsts = np.reshape(fcst, (fcst.shape[0] * fcst.shape[1], 1))
    obss = np.reshape(obs, (obs.shape[0] * obs.shape[1], 1))

    for i in range(len(fcsts)):
        if fcsts[i, 0] >= thr and obss[i, 0] >= thr:
            a += 1
        elif fcsts[i, 0] >= thr and obss[i, 0] < thr:
            b += 1
        elif fcsts[i, 0] < thr and obss[i, 0] >= thr:
            c += 1
        else:
            d += 1
    csi = CSI(a, b, c)
    far = FAR(a, b)
    pod = POD(a, c)
    acc = ACC(a, b, c, d)

    return csi, far, pod, acc