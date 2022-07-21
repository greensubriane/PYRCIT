from rcit.util.post_process.gradient import *
from rcit.util.pyramid import *
import numpy as np
import scipy.ndimage as ndi


def lukas_kanade(img0, img1, N):
    """
    
    :param img0: first radar image
    :param img1: second radar image (t+delta_t)
    :param N: size of the window (no. of equations= N**2)
    :return: flow a,b and gradients fx, fy, ft
    """

    # Iitializing flow with zero matrix
    a = np.zeros((img0.shape))
    b = np.zeros((img1.shape))

    # Calculating gradients
    fx, fy, ft = grad_cal(img0, img1)

    for x in range(N // 2, img0.shape[0] - N // 2):
        for y in range(N // 2, img0.shape[1] - N // 2):
            # Selecting block(Window) around the level
            block_fx = fx[x - N // 2:x + N // 2 + 1, y - N // 2:y + N // 2 + 1]
            block_fy = fy[x - N // 2:x + N // 2 + 1, y - N // 2:y + N // 2 + 1]
            block_ft = ft[x - N // 2:x + N // 2 + 1, y - N // 2:y + N // 2 + 1]

            # Flattening to generate equations
            block_ft = block_ft.flatten()
            block_fy = block_fy.flatten()
            block_fx = block_fx.flatten()

            # Reshaping to generate the format of Ax=B
            B = -1 * np.asarray(block_ft)
            A = np.asarray([block_fx, block_fy]).reshape(-1, 2)

            # Solving equations using pseudo inverse
            flow = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(A), A)), np.transpose(A)), B)

            # Updating flow matrix a, b
            a[x, y] = flow[0]
            b[x, y] = flow[1]

    return [a, b], [fx, fy, ft]


def horn_schunk(img0, img1, lambada, max_iter, epsilon):
    """

    :param img0: first radar image
    :param img1: second radar image(t+delta_t)
    :param lambada: hyper parameter
    :param max_iter: number of iterations
    :param epsilon: decay rate
    :return: flow and gradient
    """
    decay = 10000
    i = 0

    # averaging kernel
    avg_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4

    # calculating gradient
    fx, fy, ft = grad_cal(img0, img1)

    a = np.zeros((img0.shape))
    b = np.zeros((img0.shape))

    while (decay > epsilon and i <= max_iter):
        i += 1
        # calculating
        a_avg = ndi.convolve(input=a, weights=avg_kernel)
        b_avg = ndi.convolve(input=b, weights=avg_kernel)

        temp = (fx * a_avg + fy * b_avg + ft) / (1 + lambada * (fx ** 2 + fy ** 2))

        # updating flow
        a = a_avg - lambada * fx * temp
        b = b_avg - lambada * fy * temp

        # calculating decay
        decay = np.max(np.max((abs(a - a_avg) + abs(b - b_avg))))

    return [a, b], [fx, fy, ft]


def iterative_lucas_kanade(img0, img1, N, old_flow):
    a_old = old_flow[0]
    b_old = old_flow[1]
    fx, fy, ft = grad_cal(img0, img1)

    # wraping image with old flow
    pred = np.round((img0/255 + fx*a_old + fy*b_old +ft)*255)
    # pred = np.round((img0 + fx * a_old + fy * b_old + ft))
    pred[pred>255]=255

    # calculatng al~ and bl~
    flow, grad = lukas_kanade(pred, img1, N)

    # new flow calculating
    new_a = a_old + flow[0]
    new_b = b_old + flow[1]

    return [new_a, new_b]


def multiscale_lukas_kanade(img0, img1, N, levels):
    pyr0, shapes0 = pyramid_down(img0, levels)
    pyr2, shapes1 = pyramid_down(img1, levels)

    # calculating initial flow at lowest scale
    [a, b], grad = lukas_kanade(pyr0[0:shapes0[levels - 1][0], 0:shapes0[levels - 1][1], levels - 1],
                                pyr2[0:shapes0[levels - 1][0], 0:shapes0[levels - 1][1], levels - 1], N)

    # upsample flow for next level
    a2 = cv2.pyrUp(a)
    b2 = cv2.pyrUp(b)

    for i in range(levels - 2, -1, -1):
        [a, b] = iterative_lucas_kanade(pyr0[0:shapes0[i][0], 0:shapes0[i][1], i],
                                        pyr2[0:shapes0[i][0], 0:shapes0[i][1], i], N, [a2, b2])

        # upsample flow for next level
        a2 = cv2.pyrUp(a)
        b2 = cv2.pyrUp(b)

    grad = grad_cal(img0, img1)
    return [a, b], grad


def iterative_horn_schunk(img0, img1, old_flow, lambada, max_iter, epsilon):
    a_old = old_flow[0]
    b_old = old_flow[1]
    fx, fy, ft = grad_cal(img0, img1)

    # wraping radar image with old flow
    pred = np.round((img0 / 255 + fx * a_old + fy * b_old + ft) * 255)
    pred[pred > 255] = 255

    #pred = np.round((img0 + fx * a_old + fy * b_old + ft))

    # Calculating a1~ and b1~
    flow, grad = horn_schunk(pred, img1, lambada, max_iter, epsilon)

    # New flow
    new_a = a_old + flow[0]
    new_b = b_old + flow[1]

    return [new_a, new_b]


def multiscale_horn_schunk_flow(img0, img1, lambada, max_iter, epsilon, levels):
    pyr0, shapes0 = pyramid_down(img0, levels)
    pyr2, shapes2 = pyramid_down(img1, levels)

    # Calculate initial flow at lowest scale
    [a, b], grad = horn_schunk(pyr0[0:shapes0[levels - 1][0], 0:shapes0[levels - 1][1], levels - 1],
                               pyr2[0:shapes0[levels - 1][0], 0:shapes0[levels - 1][1], levels - 1],
                               lambada, max_iter,
                               epsilon)

    # upsample flow for next level
    a2 = cv2.pyrUp(a)
    b2 = cv2.pyrUp(b)

    for i in range(levels - 2, -1, -1):
        [a, b] = iterative_horn_schunk(pyr0[0:shapes0[i][0], 0:shapes0[i][1], i],
                                       pyr2[0:shapes0[i][0], 0:shapes0[i][1], i],
                                       [a2, b2],
                                       lambada,
                                       max_iter,
                                       epsilon)

        # upsample flow for next level
        a2 = cv2.pyrUp(a)
        b2 = cv2.pyrUp(b)

    grad = grad_cal(img0, img1)
    return [a, b], grad
