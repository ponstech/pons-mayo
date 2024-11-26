import os

import numpy as np
import cv2
from Boundcon import Boundcon
from transmissionMap import CalTransmission
from dehazeFunction import Dehazefun
from phaseaymmono import phaseasymmono


def image_enhancement(imageUS, threshold=0.7):
    if imageUS.shape[2] == 4:
        imageUS = imageUS[:, :, :3]

    imageUS = cv2.cvtColor(imageUS, cv2.COLOR_RGB2GRAY)
    # imageUS = (imageUS - np.min(imageUS)) / (np.max(imageUS) - np.min(imageUS)) * 255
    imageUS = imageUS.astype(np.uint8)

    # HazeImg = np.stack((imageUS, imageUS, imageUS), axis=-1)
    # A = np.array([threshold * np.max(imageUS)] * 6)
    # # wsz = 3 previous parameter value
    # wsz = 20
    # ts = Boundcon(HazeImg, A, 30, 300, wsz)

    # lambda_ = 2
    # t = CalTransmission(HazeImg, ts, lambda_, 5.0, max_iterations=8)
    # # %%
    # # dehazing
    # # HazeImg = mat_to_np('HazeImg')
    # #     t = mat_to_np('t')
    # r1 = Dehazefun(HazeImg, t, A, 0.85)  # working correctly
    # #     print(r1.shape)
    # r1 = np.float32(r1)
    # #     r1 = cv2.cvtColor(r1, cv2.COLOR_RGB2BGR)
    # ft, symmetryEnergy = phaseasymmono(imageUS, 2, 25, 3, 4, 'ASSD', 2, 0.2, 3, 1, 0)
    # #     np.savetxt('score.txt', symmetryEnergy, fmt = '%.4f')
    # # symmetryEnergy = mat_to_np('symmetryEnergy')

    # im2 = ft - np.amin(ft.flatten())
    # im2 = 255.0 * (im2 / np.amax(np.amax(im2)))

    # # %%
    # ilk = symmetryEnergy * ft
    # im3 = ilk - np.amin(ilk.flatten())
    # im3 = 255.0 * (im3 / np.amax(np.amax(im3)))
    # # %%
    # phase1 = 255.0 * np.divide(r1, np.max(np.max(r1)))  # abundant
    # phase2 = 255.0 * (im2 / np.amax(np.amax(im2)))
    # phase3 = 255.0 * (im3 / np.amax(np.amax(im3)))

    ft, symmetryEnergy = phaseasymmono(imageUS, 2, 25, 3, 4, 'ASSD', 2, 0.2, 3, 1, 0)

    im1 = symmetryEnergy-symmetryEnergy.min();
    im1 = 255*(im1/im1.max());

    im2 = ft-ft.min();
    im2 = 255*(im2/im2.max());

    ilk = symmetryEnergy*ft;
    im3 = ilk-ilk.min();
    im3 = 255*(im3/im3.max());

    HazeImg = np.stack((im1, im1, im1), axis=-1)
    A = np.array([threshold * np.max(im1)+1e-10] * 3).round()
    # wsz = 3 previous parameter value
    wsz = 3
    ts = Boundcon(HazeImg, A, 30, 300, wsz)

    lambda_ = 2
    t = CalTransmission(HazeImg, ts, lambda_, 0.5, max_iterations=8)
    r1 = Dehazefun(HazeImg, t, A, 0.85) # working correctly
    # r1 = cv2.cvtColor(r1, cv2.COLOR_RGB2GRAY)/255
    r1[r1>1]=1
    r1[r1<0]=0

    phase1= 255*(r1/r1.max());
    phase2= 255*(im2/im2.max());
    phase3= 255*(im3/im3.max());

    return mergeImage(phase1, phase2, phase3)


def mergeImage(phase1, phase2, phase3):
    # merged_image = np.stack((phase3, phase1[:, :, 0], phase2), axis=-1).astype(np.float32)
    merged_image = np.stack((phase3, phase1, phase2), axis=-1).astype(np.float32)
    return merged_image
