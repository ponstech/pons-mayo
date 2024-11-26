"""
The Plan:
1. Refine the transmission map (t) by ensuring its values don't get too close to zero and raising it to the power of delta.
   - This adjustment ensures that the dehazing process is stable and can control the amount of haze to be removed.
2. Convert the input hazy image's data type to float for the subsequent calculations.
3. Adjust the atmospheric light (A) to be either scalar or a 3-element array based on its input format.
4. Compute the dehazed values for the red (R), green (G), and blue (B) channels separately using the formula:
   - DehazedValue = (HazyValue - AtmosphericLight) / Transmission + AtmosphericLight
5. Combine the three channels and normalize the values to lie between [0, 1].

Parameters:
- HazeImg: A 3-channel hazy image.
- t: Transmission map estimated for the hazy image.
- A: Atmospheric light. Can be a scalar or a 3-element array.
- delta: The power to which the transmission map is raised. A higher value might result in a stronger dehazing effect.

Returns:
- rImg: The resultant dehazed image with pixel values normalized between [0, 1].

Dependencies:
- numpy: for numerical operations.
- cv2: for image processing operations.
"""
import numpy as np

# Function to display the image, does the same thing as the Dehazefunction in MATLAB

def Dehazefun(HazeImg, t, A, delta):
    t = np.maximum(np.abs(t), 0.0001) ** delta
    HazeImg = HazeImg.astype(float)
    A = np.array([A, A, A]) if np.isscalar(A) else A
    R = (HazeImg[:, :, 0] - A[0]) / t + A[0]
   #  G = (HazeImg[:, :, 1] - A[1]) / t + A[1]
   #  B = (HazeImg[:, :, 2] - A[2]) / t + A[2]
   #  rImg = np.stack((R, G, B), axis=-1)
   #  rImg = (rImg - np.min(rImg)) / (np.max(rImg) - np.min(rImg)) * 255  # Normalize pixel values
   #  rImg = rImg.astype(np.uint8)
    return R/255
    