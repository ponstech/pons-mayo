import numpy as np
import cv2
from Boundcon import Boundcon
from transmissionMap import CalTransmission
from dehazeFunction import Dehazefun
def image_quality_impr(imageUS, threshold):
    if imageUS.shape[2] == 4:
        imageUS = imageUS[:, :, :3]

    imageUS = cv2.cvtColor(imageUS, cv2.COLOR_RGB2GRAY)
    # imageUS = (imageUS - np.min(imageUS)) / (np.max(imageUS) - np.min(imageUS)) * 255
    imageUS = (imageUS - np.min(imageUS)) / np.max(imageUS) * 255
    # imageUS = imageUS.astype(np.uint8)

    HazeImg = np.stack((imageUS, imageUS, imageUS), axis=-1)
    # A = np.array([threshold * np.max(imageUS)] * 6)
    A = np.array([threshold * np.max(imageUS)+1e-10] * 3).round()

    #wsz = 3 previous parameter value
    wsz=3
    ts = Boundcon(HazeImg, A, 30, 300, wsz)

    lambda_ = 2
    # t = CalTransmission(HazeImg, ts, lambda_, 5.0, max_iterations=8)
    t = CalTransmission(HazeImg, ts, lambda_, 0.5, max_iterations=8)

    # t = t.astype(np.float32)

    # if t.size > 0:
    #     guided_filter = cv2.ximgproc.createGuidedFilter(guide=HazeImg, radius=6, eps=0.02 ** 2)
    #     t = guided_filter.filter(t)

    # r_im1 = Dehazefun(HazeImg, t, A, 0.8)

    enh = 1 - t
    # enh = (enh - np.min(enh)) / (np.max(enh) - np.min(enh)) * 255
    enh = (enh) / (np.max(enh)) * 255

    # imageUS_color = cv2.cvtColor(imageUS, cv2.COLOR_GRAY2BGR)
    # enh_color = cv2.cvtColor(enh.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # imageUS_quality_impr = cv2.add(imageUS_color, enh_color)
    # imageUS_quality_impr = np.clip(imageUS_quality_impr, 0, 255).astype(np.uint8)


    imageUS_quality_impr = enh+imageUS
    resultImg = 255 * (imageUS_quality_impr / np.max(imageUS_quality_impr))
    return resultImg
    # print(resultImg)
    # enh = 1 - t
    # enh = 255 * (enh / max(max(enh)))
    # imageUS_quality_impr = imageUS + enh
    # imageUS_quality_impr = 255 * (imageUS_quality_impr / max(max(imageUS_quality_impr)))

    # cv2.imwrite('./output/imageUS.png', imageUS)
    # cv2.imwrite('./output/ts.png', ts)
    # cv2.imwrite('./output/t.png', t)
    # cv2.imwrite('./output/1_t.png', 1 - t)
    # cv2.imwrite('./output/r_im1.png', r_im1)
    # cv2.imwrite('./output/enh.png', enh)
    # cv2.imwrite('output/final.png', resultImg)
    # show(imageUS)
    # show(ts)
    # show(t)
    # show(1 - t)
    # show(r_im1)
    # show(1 - t, cmap='hot', save=True)
    # show(resultImg)
    # print((np.array(resultImg).shape))