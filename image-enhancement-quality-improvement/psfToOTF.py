"""
The Plan:
1. Pad the input PSF with zeros to match the desired shape.
2. Circularly shift the PSF to center it.
3. Calculate the Fourier Transform to get the OTF.

Parameters:
- psf: Input Point Spread Function (usually a small matrix).
- shape: The desired shape (usually the shape of the image being processed).

Returns:
- The Optical Transfer Function (OTF) corresponding to the input PSF.

Dependencies:
- numpy: for numerical operations and data manipulations.
- scipy: for Fourier Transform and signal processing operations.

"""


import numpy as np
from scipy import fftpack



def psf2otf(psf, shape):
    psf = np.pad(psf, ((0, shape[0] - psf.shape[0]), (0, shape[1] - psf.shape[1])), mode='constant')
    for i in range(psf.ndim):
        psf = np.roll(psf, -int(psf.shape[i] / 2), axis=i)
    return fftpack.fftn(psf)