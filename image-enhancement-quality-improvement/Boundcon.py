"""
The Plan:
1. Adjust the atmospheric light (A) and the bounds (C0, C1) to have consistent dimensions.
2. Ensure the hazy image's values are in float type for calculations.
3. Compute the transmission maps for the red (t_r), green (t_g), and blue (t_b) channels.
   - For each channel, calculate the ratio of the difference between the atmospheric light and the hazy image's intensity
     to the difference between the atmospheric light and the respective bounds.
   - Take the maximum ratio value as the transmission map for that channel.
4. Combine the channel-specific transmission maps to get the maximum transmission across channels.
5. Apply a morphological closing operation using a rectangular structuring element. This step helps in removing small 
   dark spots and enhances the transmission map.

Parameters:
- HazeImg: A 3-channel hazy image.
- A: Atmospheric light vector. Can be single-valued or a 3-element array.
- C0: Lower bound for transmission. Can be single-valued or a 3-element array.
- C1: Upper bound for transmission. Can be single-valued or a 3-element array.
- sz: Size of the rectangular structuring element for morphological closing.

Returns:
- t_bdcon: The bound-constrained transmission map after morphological processing.

Dependencies:
- numpy: for numerical operations.
- scipy.ndimage: for applying the minimum filter.
- cv2: for morphological operations and structuring elements.
"""
import numpy as np
import cv2
def Boundcon(HazeImg, A, C0, C1, sz):
    if len(A) == 1:
        A = A * np.ones((3, 1))

    C0 = C0 * np.ones((3, 1))
    C1 = C1 * np.ones((3, 1))

    HazeImg = HazeImg.astype(float)
    eps = 1e-6

    t_r = np.maximum((A[0] - HazeImg[:, :, 0]) / (A[0] - C0[0] + eps), (HazeImg[:, :, 0] - A[0]) / (C1[0] - A[0] + eps))
    t_g = np.maximum((A[1] - HazeImg[:, :, 1]) / (A[1] - C0[1] + eps), (HazeImg[:, :, 1] - A[1]) / (C1[1] - A[1] + eps))
    t_b = np.maximum((A[2] - HazeImg[:, :, 2]) / (A[2] - C0[2] + eps), (HazeImg[:, :, 2] - A[2]) / (C1[2] - A[2] + eps))

    t_max = np.maximum(t_r, t_g, t_b)
    t_b = cv2.min(t_max, 1)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    t_bdcon = cv2.morphologyEx(t_b, cv2.MORPH_CLOSE, se)

    return t_bdcon