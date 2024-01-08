import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2
import math


def gaussian(x, sigma) :
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x ** 2) / (2 * (sigma ** 2)))


def bilateral(image, sigma_r, sigma_s) :
    l = 1e-10       # lambda
    minI = np.min(image) - l
    maxI = np.max(image) + l
    numSegments = int(math.ceil((maxI - minI) / sigma_r))
    
    result = np.zeros(image.shape)
    for ch in range(image.shape[2]) :
        for j in range(numSegments + 1) :
            I = image[:, :, ch]
            i = minI + j * (maxI - minI) / numSegments
            G = gaussian(I - i, sigma_r)
            K = cv2.GaussianBlur(G, (-1, -1), sigma_s)
            H = np.multiply(G, I)
            HH = cv2.GaussianBlur(H, (-1, -1), sigma_s)
            J = np.divide(HH, K)
            w = 1.0 - np.abs(I - i) / ((maxI - minI) / numSegments)
            mask = ((w < 0) | (w > 1))
            w[mask] = 0
            result[:, :, ch] += np.multiply(J, w)

    return result


def jointBilateral(imageAmbient, imageFlash, sigma_r, sigma_s) :
    l = 1e-10       # lambda
    minF = np.min(imageFlash) - l
    maxF = np.max(imageFlash) + l
    numSegments = int(math.ceil((maxF - minF) / sigma_r))
    
    result = np.zeros(imageAmbient.shape)
    for ch in range(imageAmbient.shape[2]) :
        for j in range(numSegments + 1) :
            I = imageAmbient[:, :, ch]
            F = imageFlash[:, :, ch]
            f = minF + j * (maxF - minF) / numSegments
            G = gaussian(F - f, sigma_r)
            K = cv2.GaussianBlur(G, (-1, -1), sigma_s)
            H = np.multiply(G, I)
            HH = cv2.GaussianBlur(H, (-1, -1), sigma_s)
            J = np.divide(HH, K)
            w = 1.0 - np.abs(F - f) / ((maxF - minF) / numSegments)
            mask = ((w < 0) | (w > 1))
            w[mask] = 0
            result[:, :, ch] += np.multiply(J, w)

    return result


def detailTransfer(imageNR, imageFlash, sigma_r, sigma_s) :
    epsilon = 1e-10
    F = imageFlash
    F_base = bilateral(F, sigma_r, sigma_s)
    result = imageNR * ((F + epsilon) / (F_base + epsilon))
    return result


def linearize(image) :
    mask = (image <= 0.0404482)
    result = np.array(image)
    result[mask] = result[mask] / 12.92
    mask = np.logical_not(mask)
    result[mask] = ((result[mask] + 0.055) / 1.055) ** 2.4
    return result


def createMask(imageAmbient, imageFlash, ISOAmbient, ISOFlash) :
    tau = 4 / 256
    print(tau)
    A = linearize(imageAmbient)
    F = linearize(imageFlash)
    A = A * ISOFlash / ISOAmbient
    
    L_A = (0.2126 * A[:, :, 0] + 0.7152 * A[:, :, 1] + 0.0722 * A[:, :, 2])
    L_F = (0.2126 * F[:, :, 0] + 0.7152 * F[:, :, 1] + 0.0722 * F[:, :, 2])
    M = ((L_F - L_A) <= tau)
    return M


def applyMask(base, detail, mask) :
    mask = np.stack((mask, mask, mask), axis=-1)
    result = mask * base + (1 - mask) * detail
    return result


filename1 = '../data/storage/storage_ambient.jpg'
filename2 = '../data/storage/storage_flash.jpg'
A = skimage.io.imread(filename1)
A = np.divide(A, 255.0, dtype=np.float32)
F = skimage.io.imread(filename2)
F = np.divide(F, 255.0, dtype=np.float32)
sigma_r, sigma_s = 0.4, 12
A_base = bilateral(A, sigma_r, sigma_s)
A_nr = jointBilateral(A, F, sigma_r, sigma_s)
A_detail = detailTransfer(A_nr, F, sigma_r, sigma_s)
M = createMask(A, F, 1600, 200)
A_final = applyMask(A_base, A_detail, M)
A_final = np.clip(A_final, 0.0, 1.0)

# plt.subplot(3, 3, 1)
# plt.imshow(A_base)
# plt.subplot(3, 3, 2)
# plt.imshow(A_nr)
# plt.subplot(3, 3, 4)
# plt.imshow(A_detail)
# plt.subplot(3, 3, 5)
# plt.imshow(A_final)
# plt.subplot(3, 3, 7)
# plt.imshow(np.abs(A_nr - A_base) * 10)
# plt.subplot(3, 3, 8)
# plt.imshow(np.abs(A_detail - A_base) * 10)
# plt.subplot(3, 3, 9)
# plt.imshow(np.abs(A_final - A_base) * 10)
# plt.show()

A_final = np.clip(A_final, 0, 1)
plt.imshow(A_final)
plt.show()
plt.imsave('storage_final.jpg', A_final)