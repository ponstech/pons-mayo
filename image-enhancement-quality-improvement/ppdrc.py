import cv2
import numpy as np
from scipy.fftpack import fftshift, ifftshift

from tools import rayleighmode as _rayleighmode
from tools import lowpassfilter as _lowpassfilter
from filtergrid import filtergrid
# Try and use the faster Fourier transform functions from the pyfftw module if
# available
from tools import fft2, ifft2
def highpassmonogenic(img, maxwavelength = [2,4], n = 2):
    
    eps = 1e-3

    if img.dtype not in ['float32', 'float64']:
        img = np.float64(img)
        imgdtype = 'float64'
    else:
        imgdtype = img.dtype

    if img.ndim == 3:
        img = img.mean(2)
        
    assert min(maxwavelength) >= 2, f"'Minimum wavelength must be at least  2 pixels'):"
    nscales = len(maxwavelength)
    IMG = fft2(img)

    # Generate monogenic and filter grids
    #(H1, H2, freq) = monogenicfilters(size(img))
    rows,cols = img.shape;
    radius, u1, u2 = filtergrid(rows,cols)
    
    
    # Get rid of the 0 radius value at the 0 frequency point (at top-left
    # corner after fftshift) so that taking the log of the radius will not
    # cause trouble.
    radius[0, 0] = 1.
    H = (1j * u1 - u2) / radius
    
    zeromat = np.zeros((1, nscales), dtype='object')
    
    
    
    phase = zeromat.copy()
    print("phase's shape:",phase.shape)
    orient = zeromat.copy()
    E = zeromat.copy()
    f = zeromat.copy()
    h1f = zeromat.copy()
    h2f = zeromat.copy()
    H = zeromat.copy()

    for s in range(nscales):
        
        # High pass Butterworth filter
        H =  1.0 - (1.0 / (1.0 + (np.multiply(radius,maxwavelength[s])**(2*n))))
        f = np.real(ifft2(np.multiply(H,IMG)))
        print("f_shape:",f.shape)
        h1f = np.real(ifft2(H*H[1]*IMG))
        print("******************",h1f.shape)
        h2f = np.real(ifft2(H*H[2]*IMG))
        print("h2f_shape:",h2f.shape)
        print("sqrt:",np.sqrt(np.add(h1f**2 , h2f**2) + eps ).shape)
        print("[0] and [s]", phase[0][0])
        phase[0][s] = np.arctan(f / np.sqrt(np.add(h1f**2 , h2f**2) + eps ))
        orient[0][s] = np.arctan(h2f, h1f)
        E[0][s] = np.sqrt(f**2 + h1f**2 + h2f**2)
    

    # If a single scale specified return output matrices directly
    if nscales == 1:
        return phase[1], orient[1], E[1]
    else:
        
        return phase, orient, E



im = cv2.imread('img.png')
img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 

phase,orient,E = highpassmonogenic(img = img)

cv2.imshow("w",orient)

cv2.waitKey(0)
cv2.destroyAllWindows()