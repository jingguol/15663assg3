import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2
from scipy import signal
import math


def gradient(I) :
    # axis0 is x-axis, axis1 is y-axis
    I_x = np.diff(I, axis=0, prepend=0)
    I_y = np.diff(I, axis=1, prepend=0)
    return I_x, I_y


def divergence(I_x, I_y) :
    I_xx, I_xy = gradient(I_x)
    I_yx, I_yy = gradient(I_y)
    return I_xx + I_yy


def laplacian(I) :
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    result = np.zeros(I.shape)
    if I.ndim == 2 :
        result = signal.convolve2d(I, kernel, mode='same', boundary='fill', fillvalue=0)
    else :
        for i in range(I.shape[2]) :
            result[:, :, i] = signal.convolve2d(I[:, :, i], kernel, mode='same', boundary='fill', fillvalue=0)
    return result


def conjugateGradientDescent(D, I_init, B, I_boundary, epsilon, N) :
    I = B * I_init + (1 - B) * I_boundary
    r = B * (D - laplacian(I))
    d = np.array(r)
    delta_new = np.sum(r * r)
    n = 0

    while delta_new > epsilon ** 2 and n < N :
        if n % 10 == 0 :
            print('iteration ' + str(n))
        q = laplacian(d)
        eta = delta_new / np.sum(d * q)
        I = I + B * (eta * d)
        r = B * (r - eta * q)
        delta_old = delta_new
        delta_new = np.sum(r * r)
        beta = delta_new / delta_old
        d = r + beta * d
        n = n + 1

    return I


filename1 = '../data/book/book_ambient.jpg'
filename2 = '../data/book/book_flash.jpg'
A = skimage.io.imread(filename1)
A = np.divide(A, 255.0, dtype=np.float32)[:, :, 0:3]
F = skimage.io.imread(filename2)
F = np.divide(F, 255.0, dtype=np.float32)[:, :, 0:3]

A_x, A_y = gradient(A)
F_x, F_y = gradient(F)
M = (np.absolute(F_x * A_x + F_y * A_y)) / (np.sqrt(F_x ** 2 + F_y ** 2) * np.sqrt(A_x ** 2 + A_y ** 2) + 1e-10)

sigma = 160
tau_s = 0.5
w_s = np.tanh(sigma * (F - tau_s))
w_s = w_s / (np.max(w_s) - np.min(w_s)) + np.min(w_s)

I_x = w_s * A_x + (1 - w_s) * (M * F_x + (1 - M) * A_x)
I_y = w_s * A_y + (1 - w_s) * (M * F_y + (1 - M) * A_y)

# plt.subplot(1, 2, 1)
# plt.imshow(A_x / np.max(A_x) + 0.5)
# plt.subplot(1, 2, 2)
# plt.imshow(A_y / np.max(A_y) + 0.5)
# plt.show()
# plt.subplot(1, 2, 1)
# plt.imshow(F_x / np.max(F_x) + 0.5)
# plt.subplot(1, 2, 2)
# plt.imshow(F_y / np.max(F_y) + 0.5)
# plt.show()
# plt.subplot(1, 2, 1)
# plt.imshow(I_x / np.max(I_x) + 0.5)
# plt.subplot(1, 2, 2)
# plt.imshow(I_y / np.max(I_y) + 0.5)
# plt.show()

D = divergence(I_x, I_y)
# I_init = np.zeros(A.shape)
I_init = A
B = np.ones(A.shape)
B[0, :, :] = 0
B[-1, :, :] = 0
B[:, 0, :] = 0
B[:, -1, :] = 0
epsilon = 1e-2
N = 500
I = conjugateGradientDescent(D, I_init, B, F, epsilon, N)

I = np.clip(I, 0, 1)
plt.imshow(I)
plt.show()
plt.imsave('book_fused.jpg', I)