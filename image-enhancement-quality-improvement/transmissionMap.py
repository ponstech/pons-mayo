"""
The Plan:
1. CalWeightFun: 
    - Take in the hazy image, a directional filter `D`, and a parameter `sigma`.
    - Compute weighted functions for each color channel using convolution with the directional filter. 
    - This function  measures the local changes in intensity/color to guide the transmission estimation.

2. CalTransmission:
    - Usee the Fast Fourier Transform (FFT) for efficient computation.
    - Take the hazy image, an initial estimate of transmission `t`, a regularization parameter `lambda_`, a parameter `param`, and maximum number of iterations.
    - Use a set of pre-defined directional filters to compute the transmission map. 
    - Iteratively refine the transmission map using an optimization approach, updating the map with respect to the weighted functions and constraints.

Dependencies:
    - Requires numpy for numerical operations.
    - Uses functions from scipy's signal module for 2D convolutions.
    - Uses scipy's fftpack for Fast Fourier Transforms.
"""
import numpy as np
from scipy import signal, fftpack
import cv2
def CalWeightFun(HazeImg, D, param):
    sigma = param
    HazeImg = HazeImg.astype(float) / 255
    d_r = signal.convolve2d(HazeImg[:, :, 0], D, mode='same')
    d_g = signal.convolve2d(HazeImg[:, :, 1], D, mode='same')
    d_b = signal.convolve2d(HazeImg[:, :, 2], D, mode='same')
    WFun = np.exp(-(d_r**2 + d_g**2 + d_b**2) / sigma / 2)
    return WFun

def CalTransmission(HazeImg, t, lambda_, param,max_iterations=6):
    nRows, nCols = t.shape
    nsz = 3
    NUM = nsz * nsz
    D = [np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
         np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
         np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
         np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
         np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
         np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
         np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
         np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])]
    D = [d / np.linalg.norm(d) for d in D]
    WFun = [CalWeightFun(HazeImg, d, param) for d in D]
    Tf = fftpack.fftn(t)
    DS = sum([abs(fftpack.fftn(d, shape=(nRows, nCols)))**2 for d in D])

    beta = 1
    beta_rate = 2 * np.sqrt(2)
    max_beta = 2**8
    Outiter = 0

    while Outiter < max_iterations and beta < max_beta:
        gamma = lambda_ / beta
        Outiter += 1
        # print(f'Outer iteration {Outiter}; beta {beta:.3g}')
        DU = 0
        for d, w in zip(D, WFun):
            dt = signal.convolve2d(t, d, mode='same', boundary='wrap')
            u = np.maximum(abs(dt) - w / beta / len(D), 0) * np.sign(dt)
            DU += fftpack.fftn(signal.convolve2d(u, np.rot90(d, 2), mode='same', boundary='wrap'))
        t = np.abs(fftpack.ifftn((gamma * Tf + DU) / (gamma + DS)))
        beta *= beta_rate

    return t
