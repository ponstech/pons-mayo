# PHASEASYMMONO - phase Asymmetry of an image using monogenic filters
#
# This function calculates the phase symmetry of points in an image.
# This is a contrast invariant measure of symmetry.  This function can be
# used as a line and blob detector.  The greyscale 'polarity' of the lines
# that you want to find can be specified.
#
# This code is considerably faster than PHASESYM but you may prefer the
# output from PHASESYM's oriented filters.
#
# There are potentially many arguments, here is the full usage:
#   [phaseSym or ft T symmetryEnergy] =  ...
#                phasecongmono(im, nscale, minWaveLength, mult, ...
#                         bandwidth, filter, shape, alpha, k, polarity, cutOff, g, noiseMethod)
#
# However, apart from the image, all parameters have defaults and the
# usage can be as simple as:
#
#    phaseSym = phaseasymmono(im);
#
# Arguments:
#              Default values      Description
#
#    nscale           5    - Number of wavelet scales, try values 3-6
#    minWaveLength    3    - Wavelength of smallest scale filter.
#    mult             2.1  - Scaling factor between successive filters.
#    bandwidth        2    - Bandwidth in octaves of the filter or Ratio of the standard
#                            deviation of the Gaussian describing the log Gabor
#                            filter's transfer function in the frequency domain
#                            to the filter center frequency.
#---------------------------
#    filter                - Type of filter, LogGabor, SSD or DoSS
#    shape                 - Parameter of the filters: derivative parameter
#                            a in N+ for SSD, and shape parameter gama in [0, 1]
#                            for DoSS
#    alpha                 - Alpha parameter for scale spaces alpha in [0, 1]
#---------------------------------
#    k                2.0  - No of standard deviations of the noise energy beyond
#                            the mean at which we set the noise threshold point.
#                            You may want to vary this up to a value of 10 or
#                            20 for noisy images
#    polarity         0    - Controls 'polarity' of symmetry features to find.
#                             1 - just return 'bright' points
#                            -1 - just return 'dark' points
#                             0 - return bright and dark points.
#    noiseMethod      -1   - Parameter specifies method used to determine
#                            noise statistics.
#                              -1 use median of smallest scale filter responses
#                              -2 use mode of smallest scale filter responses
#                               0+ use noiseMethod value as the fixed noise threshold
#                            A value of 0 will turn off all noise compensation.
#
# Return values:
#    phaseSym              - Phase symmetry image (values between 0 and 1).
#    symmetryEnergy        - Un-normalised raw symmetry energy which may be
#                            more to your liking.
#    T                     - Calculated noise threshold (can be useful for
#                            diagnosing noise characteristics of images)
#
#
# Notes on specifying parameters:
#
# The parameters can be specified as a full list eg.
#  >> phaseSym = phasesym(im, 5, 3, 2.5, 0.55, 2.0, 0);
#
# or as a partial list with unspecified parameters taking on default values
#  >> phaseSym = phasesym(im, 5, 3);
#
# or as a partial list of parameters followed by some parameters specified via a
# keyword-value pair, remaining parameters are set to defaults, for example:
#  >> phaseSym = phasesym(im, 5, 3, 'polarity',-1, 'k', 2.5);
#
# The convolutions are done via the FFT.  Many of the parameters relate to the
# specification of the filters in the frequency plane.  The values do not seem
# to be very critical and the defaults are usually fine.  You may want to
# experiment with the values of 'nscales' and 'k', the noise compensation factor.
#
# Notes on filter settings to obtain even coverage of the spectrum
# sigmaOnf       .85   mult 1.3
# sigmaOnf       .75   mult 1.6     (filter bandwidth ~1 octave)
# sigmaOnf       .65   mult 2.1
# sigmaOnf       .55   mult 3       (filter bandwidth ~2 octaves)
#
# For maximum speed the input image should have dimensions that correspond to
# powers of 2, but the code will operate on images of arbitrary size.
#
# See Also:  PHASESYM, PHASECONGMONO
import cv2
# References:
#     Peter Kovesi, "Symmetry and Asymmetry From Local Phase" AI'97, Tenth
#     Australian Joint Conference on Artificial Intelligence. 2 - 4 December
#     1997. http://www.cs.uwa.edu.au/pub/robvis/papers/pk/ai97.ps.gz.
#
#     Peter Kovesi, "Image Features From Phase Congruency". Videre: A
#     Journal of Computer Vision Research. MIT Press. Volume 1, Number 3,
#     Summer 1999 http://mitpress.mit.edu/e-journals/Videre/001/v13.html
#
#     Michael Felsberg and Gerald Sommer, "A New Extension of Linear Signal
#     Processing for Estimating Local Properties and Detecting Features". DAGM
#     Symposium 2000, Kiel
#
#     Michael Felsberg and Gerald Sommer. "The Monogenic Signal" IEEE
#     Transactions on Signal Processing, 49(12):3136-3144, December 2001



# July 2008      Code developed from phasesym where local phase information
#                calculated using Monogenic Filters.
# April 2009     Noise compensation simplified to speedup execution.
#                Options to calculate noise statistics via median or mode of
#                smallest filter response.  Option to use a fixed threshold.
#                Return of T for 'instrumentation' of noise detection/compensation.
#                Removal of orientation calculation from phasesym (not clear
#                how best to calculate this from monogenic filter outputs)
# June 2009      Clean up

# Copyright (c) 1996-2009 Peter Kovesi
# School of Computer Science & Software Engineering
# The University of Western Australia
# pk at csse uwa edu au
# http://www.csse.uwa.edu.au/
#
# Permission is hereby  granted, free of charge, to any  person obtaining a copy
# of this software and associated  documentation files (the "Software"), to deal
# in the Software without restriction, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# The software is provided "as is", without warranty of any kind.
#
#------------------------------------------------------------------------
#
# Modified by A. BELAID, 2013
#
# Description: This code implements a part of the paper:
# Ahror BELAID and Djamal BOUKERROUI. "A new generalised alpha scale
# spaces quadrature filters." Pattern Recognition 47.10 (2014): 3209-3224.
#
# Ahror BELAID and Djamal BOUKERROUI. "Alpha scale space filters for
# phase based edge detection in ultrasound images." ISBI (2014): 1247-1250.
#
# Copyright (c), Heudiasyc laboratory, Compi�gne, France.
#
#------------------------------------------------------------------------

import numpy as np
from tools import fft2,ifftshift,ifft2,rayleighmode
from low_pass_filter import lowpassfilter
from ASSD_get_param import ASSD_get_param
from cv2 import COLOR_RGB2GRAY, cvtColor
# from PIL import Image, ImageOps

def phaseasymmono(im, nscale = 5, minWaveLength = 3, mult = 2.1, bandwidth = 2.5, filter = "ASSD", shape = 2, alpha = 0.5, k = 2.0, noiseMethod = 0, cutOff = 0.5, g = 10, polarity = 1): 
    # Get arguments and/or default values
    #im,nscale,minWaveLength,mult,bandwidth,filter,shape,alpha,k,noiseMethod,cutOff,g,polarity = checkargs(varargin)
    #    filter= 'LG';

    
    shape = im.shape
    
    epsilon = 0.0001
    if len(shape) == 3:
        # img = Image.fromarray(im)
        img = cv2.cvtColor(im, COLOR_RGB2GRAY)
        # img = ImageOps.grayscale(img)

        img = np.array(img)
    
    rows,cols= im.shape
    im0 = im
    
    IM = fft2(im)
    
    
    zero = np.zeros((rows,cols))
    symmetryEnergy = zero
    
    # symmetry values (energy).
    sumAn = zero
    
    # amplitude values.
    sumf = zero
    sumh1 = zero
    sumh2 = zero
    # Pre-compute some stuff to speed up filter construction
    
    # Set up u1 and u2 matrices with ranges normalised to +/- 0.5
# The following code adjusts things appropriately for odd and even values
# of rows and columns.
    if np.mod(cols,2):
        xrange = np.array([np.arange(- (cols - 1) / 2,(cols - 1) / 2+1)]) / (cols - 1)
    else:
        xrange = np.array([np.arange(- cols / 2,(cols / 2 - 1)+1)]) / cols
    
    if np.mod(rows,2):
        yrange = np.array([np.arange(- (rows - 1) / 2,(rows - 1) / 2+1)]) / (rows - 1)
    else:
        yrange = np.array([np.arange(- rows / 2,(rows / 2 - 1)+1)]) / rows
    
    u1,u2 = np.meshgrid(xrange,yrange)
    u1 = ifftshift(u1)
    
    
    u2 = ifftshift(u2)
    
    radius = np.sqrt(u1 ** 2 + u2 ** 2)
    
    
    # values as a radius from centre
# (but quadrant shifted)
    
    # Get rid of the 0 radius value in the middle (at top left corner after
# fftshifting) so that taking the log of the radius, or dividing by the
# radius, will not cause trouble.
    
    radius[0,0] = 1
    # Construct the monogenic filters in the frequency domain.  The two
# filters would normally be constructed as follows
#    H1 = i*u1./radius;
#    H2 = i*u2./radius;
# However the two filters can be packed together as a complex valued
# matrix, one in the real part and one in the imaginary part.  Do this by
# multiplying H2 by i and then adding it to H1 (note the subtraction
# because i*i = -1).  When the convolution is performed via the fft the
# real part of the result will correspond to the convolution with H1 and
# the imaginary part with H2.  This allows the two convolutions to be
# done as one in the frequency domain, saving time and memory.
    
    H = np.divide(1j * u1 - u2,radius)
    
    # The two monogenic filters H1 and H2 are not selective in terms of the
# magnitudes of the frequencies.  The code below generates bandpass
# log-Gabor filters which are point-wise multiplied by IM to produce
# different bandpass versions of the image before being convolved with H1
# and H2
    
    # First construct a low-pass filter that is as large as possible, yet falls
# away to zero at the boundaries.  All filters are multiplied by
# this to ensure no extra frequencies at the 'corners' of the FFT are
# incorporated as this can upset the normalisation process when
# calculating phase symmetry
    lp = lowpassfilter(np.array([rows,cols]),0.4,10)
    
    
    for s in np.arange(1,nscale+1).reshape(-1):
        wavelength = minWaveLength * mult ** (s - 1)
        fo = 1.0 / wavelength
        if np.array(['LG']) == filter:
            sigmaOnf = np.exp(- bandwidth * np.sqrt(2 * np.log(2)) / 4)
            Filter = np.exp((- (np.log(radius / fo)) ** 2) / (2 * np.log(sigmaOnf) ** 2))
            print('LogGabor filter')
        else:
            if np.array(['ASSD']) == filter:
                if bandwidth==0:
                    beta,a = ASSD_get_param([],shape,alpha)
                else:
                    #a = ASSD_get_param(beta = bandwidth,alpha = alpha)
                    a = 1.838 # i tried with different images and found value of a does not depend on the picture, it is costant
                    
                sigma = np.divide((a / 2 / alpha) ** (1 / 2 / alpha),fo)
                nc = 1 / (np.multiply((fo ** a),np.exp(- (np.multiply(fo,sigma)) ** (2 * alpha))))
                Filter = np.multiply(np.multiply(nc,(radius ** a)),np.exp(- (np.multiply(radius,sigma)) ** (2 * alpha)))
                #Filter = Filter./max(Filter(:));
                #print(np.array(['ASSD filter, Centre frequency = ',num2str(fo,2),', sigma = ',num2str(sigma,2)]))
            
        Filter = np.multiply(Filter,lp)
        
        Filter[0,0] = 0
        
        # back to zero (undo the radius fudge).
        IMF = IM*Filter
        

        f = np.real(ifft2(IMF))
        
        h = ifft2(IMF*H)
        
        # convolution result with h1, imaginary part
# contains convolution result with h2.
        h1 = np.real(h)
        h2 = np.imag(h)
        
        sumh1 = sumh1 + h1
        sumh2 = sumh2 + h2
        sumf = sumf + f
        hAmp2 = h1 ** 2 + h2 ** 2
        
        sumAn = sumAn + np.sqrt(f ** 2 + hAmp2)
        
        # Now calculate the phase symmetry measure.
        
        if polarity == 0:
            symmetryEnergy = symmetryEnergy + np.abs(f) - np.sqrt(hAmp2)
        else:
            if polarity == 1:
                symmetryEnergy = symmetryEnergy + f - np.sqrt(hAmp2)
            else:
                if polarity == - 1:
                    symmetryEnergy = symmetryEnergy - f - np.sqrt(hAmp2)
                else:
                    if polarity == 10:
                        symmetryEnergy = symmetryEnergy - np.abs(f) + np.sqrt(hAmp2)
                        
                    else:
                        if polarity == 11:
                            symmetryEnergy = symmetryEnergy - f + np.sqrt(hAmp2)
                        else:
                            if polarity == - 11:
                                symmetryEnergy = symmetryEnergy + f + np.sqrt(hAmp2)
        # At the smallest scale estimate noise characteristics from the
# distribution of the filter amplitude responses stored in sumAn.
# tau is the Rayleigh parameter that is used to specify the
# distribution.
        #if s == 1:
         #   if noiseMethod == - 1:
               # tau = np.median(sumAn) / np.sqrt(np.log(4))
          #  else:
           #     if noiseMethod == - 2:
            #        tau = rayleighmode(sumAn)
    
    # Compensate for noise
    
    # Assuming the noise is Gaussian the response of the filters to noise will
# form Rayleigh distribution.  We use the filter responses at the smallest
# scale as a guide to the underlying noise level because the smallest scale
# filters spend most of their time responding to noise, and only
# occasionally responding to features. Either the median, or the mode, of
# the distribution of filter responses can be used as a robust statistic to
# estimate the distribution mean and standard deviation as these are related
# to the median or mode by fixed constants.  The response of the larger
# scale filters to noise can then be estimated from the smallest scale
# filter response according to their relative bandwidths.
    
    # This code assumes that the expected reponse to noise on the phase symmetry
# calculation is simply the sum of the expected noise responses of each of
# the filters.  This is a simplistic overestimate, however these two
# quantities should be related by some constant that will depend on the
# filter bank being used.  Appropriate tuning of the parameter 'k' will
# allow you to produce the desired output. (though the value of k seems to
# be not at all critical)
    
    #if noiseMethod >= 0:
        #T = noiseMethod
    #else:
        # Estimate the effect of noise on the sum of the filter responses as
# the sum of estimated individual responses (this is a simplistic
# overestimate). As the estimated noise response at succesive scales
# is scaled inversely proportional to bandwidth we have a simple
# geometric sum.
        #totalTau = tau * (1 - (1 / mult) ** nscale) / (1 - (1 / mult))
        # Calculate mean and std dev from tau using fixed relationship
# between these parameters and tau. See
# http://mathworld.wolfram.com/RayleighDistribution.html
        #EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2)
        #EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2)
        # Noise threshold, make sure it is not less than epsilon

        #T = np.amax(EstNoiseEnergyMean + k * EstNoiseEnergySigma,epsilon)
    
    # Apply noise threshold - effectively wavelet denoising soft thresholding
# and normalize symmetryEnergy by the sumAn to obtain phase symmetry.
# Note the max operation is not necessary if you are after speed, it is
# just 'tidy' not having -ve symmetry values
    #phaseSym = np.amax(symmetryEnergy - T,zero) / (sumAn + epsilon)
    #or_ = np.arctan(- sumh2 / sumh1)
    
    #or_[or_ < 0] = or_(or_ < 0) + np.pi
    
    # orientation values now range 0 - pi
    #or_ = np.rint(or_ / np.pi * 180)
    
    ft = np.arctan2(sumf,np.sqrt(sumh1 ** 2 + sumh2 ** 2))

    return ft,symmetryEnergy
    
    # -pi/2 to pi/2.
    
    #------------------------------------------------------------------
# CHECKARGS
    
    # Function to process the arguments that have been supplied, assign
# default values as needed and perform basic checks.
    
    '''
def checkargs(arg = None): 
    nargs = len(arg)
    if nargs < 1:
        raise Exception('No image supplied as an argument')
    
    # Set up default values for all arguments and then overwrite them
# with with any new values that may be supplied
    im = []
    nscale = 5
    
    minWaveLength = 3
    
    mult = 2.1
    
    bandwidth = 2.5
    
    # Gaussian describing the log Gabor filter's
# transfer function in the frequency domain
# to the filter center frequency.
    filter = 'ASSD'
    
    shape = []
    
    alpha = 0.5
    
    k = 2.0
    
    # energy beyond the mean at which we set the
# noise threshold point.
    
    noiseMethod = - 1
    
    # filter to estimate noise statistics
    cutOff = 0.5
    g = 10
    polarity = 10
    # Allowed argument reading states
    allnumeric = 1
    
    keywordvalue = 2
    
    # followed by numeric value
    readstate = allnumeric
    
    if readstate == allnumeric:
        for n in np.arange(1,nargs+1).reshape(-1):
            #       if isa(arg{n},'char')
#           readstate = keywordvalue;
#           break;
#       else
            if n == 1:
                im = arg[n]
            else:
                if n == 2:
                    nscale = arg[n]
                else:
                    if n == 3:
                        minWaveLength = arg[n]
                    else:
                        if n == 4:
                            mult = arg[n]
                        else:
                            if n == 5:
                                bandwidth = arg[n]
                            else:
                                if n == 6:
                                    filter = arg[n]
                                else:
                                    if n == 7:
                                        shape = arg[n]
                                    else:
                                        if n == 8:
                                            alpha = arg[n]
                                        else:
                                            if n == 9:
                                                k = arg[n]
                                            else:
                                                if n == 11:
                                                    noiseMethod = arg[n]
                                                else:
                                                    if n == 12:
                                                        cutOff = arg[n]
                                                    else:
                                                        if n == 13:
                                                            g = arg[n]
                                                        else:
                                                            if n == 10:
                                                                polarity = arg[n]
            # end
    
    # Code to handle parameter name - value pairs
    if readstate == keywordvalue:
        while n < nargs:

            if not True  or not True :
                raise Exception('There should be a parameter name - value pair')
            if strncmpi(arg[n],'im',2):
                im = arg[n + 1]
            else:
                if strncmpi(arg[n],'nscale',2):
                    nscale = arg[n + 1]
                else:
                    if strncmpi(arg[n],'minWaveLength',2):
                        minWaveLength = arg[n + 1]
                    else:
                        if strncmpi(arg[n],'mult',2):
                            mult = arg[n + 1]
                        else:
                            if strncmpi(arg[n],'bandwidth',2):
                                bandwidth = arg[n + 1]
                            else:
                                if strncmpi(arg[n],'filter',2):
                                    filter = arg[n + 1]
                                else:
                                    if strncmpi(arg[n],'shape',2):
                                        shape = arg[n + 1]
                                    else:
                                        if strncmpi(arg[n],'alpha',2):
                                            alpha = arg[n + 1]
                                        else:
                                            if strncmpi(arg[n],'k',1):
                                                k = arg[n + 1]
                                            else:
                                                if strncmpi(arg[n],'noisemethod',3):
                                                    noiseMethod = arg[n + 1]
                                                else:
                                                    if strncmpi(arg[n],'cutOff',2):
                                                        cutOff = arg[n + 1]
                                                    else:
                                                        if strncmpi(arg[n],'g',1):
                                                            g = arg[n + 1]
                                                        else:
                                                            if strncmpi(arg[n],'polarity',3):
                                                                polarity = arg[n + 1]
                                                            else:
                                                                raise Exception('Unrecognised parameter name')
            n = n + 2
            if n == nargs:
                raise Exception('Unmatched parameter name - value pair')

    
    if len(im)==0:
        raise Exception('No image argument supplied')
    
    if not True :
        im = double(im)
    
    if nscale < 1:
        raise Exception('nscale must be an integer >= 1')
    
    if minWaveLength < 2:
        raise Exception('It makes little sense to have a wavelength < 2')
    
    if polarity != - 1 and polarity != 0 and polarity != 1 and polarity != - 11 and polarity != 10 and polarity != 11:
        raise Exception('Allowed polarity values are -1, 0 and 1')
    
    #------------------------------------------------------------------
# CHECKARGSOLD  NOT USED
    
    # Function to process the arguments that have been supplied, assign
# default values as needed and perform basic checks.
    
    
def checkargsold(arg = None): 
    nargs = len(arg)
    if nargs < 1:
        raise Exception('No image supplied as an argument')
    
    # Set up default values for all arguments and then overwrite them
# with with any new values that may be supplied
    im = []
    nscale = 5
    
    minWaveLength = 3
    
    mult = 2.1
    
    sigmaOnf = 0.55
    
    # Gaussian describing the log Gabor filter's
# transfer function in the frequency domain
# to the filter center frequency.
    k = 2.0
    
    # energy beyond the mean at which we set the
# noise threshold point.
    
    polarity = 0
    
    noiseMethod = - 1
    
    # filter to estimate noise statistics
    
    # Allowed argument reading states
    allnumeric = 1
    
    keywordvalue = 2
    
    # followed by numeric value
    readstate = allnumeric
    
    if readstate == allnumeric:
        for n in np.arange(1,nargs+1).reshape(-1):
            if True:
                readstate = keywordvalue
                break
            else:
                if n == 1:
                    im = arg[n]
                else:
                    if n == 2:
                        nscale = arg[n]
                    else:
                        if n == 3:
                            minWaveLength = arg[n]
                        else:
                            if n == 4:
                                mult = arg[n]
                            else:
                                if n == 5:
                                    sigmaOnf = arg[n]
                                else:
                                    if n == 6:
                                        k = arg[n]
                                    else:
                                        if n == 7:
                                            polarity = arg[n]
                                        else:
                                            if n == 8:
                                                noiseMethod = arg[n]
    
    # Code to handle parameter name - value pairs
    if readstate == keywordvalue:
        while n < nargs:

            if not True  or not True :
                raise Exception('There should be a parameter name - value pair')
            if strncmpi(arg[n],'im',2):
                im = arg[n + 1]
            else:
                if strncmpi(arg[n],'nscale',2):
                    nscale = arg[n + 1]
                else:
                    if strncmpi(arg[n],'minWaveLength',2):
                        minWaveLength = arg[n + 1]
                    else:
                        if strncmpi(arg[n],'mult',2):
                            mult = arg[n + 1]
                        else:
                            if strncmpi(arg[n],'sigmaOnf',2):
                                sigmaOnf = arg[n + 1]
                            else:
                                if strncmpi(arg[n],'k',1):
                                    k = arg[n + 1]
                                else:
                                    if strncmpi(arg[n],'polarity',2):
                                        polarity = arg[n + 1]
                                    else:
                                        if strncmpi(arg[n],'noisemethod',3):
                                            noiseMethod = arg[n + 1]
                                        else:
                                            raise Exception('Unrecognised parameter name')
            n = n + 2
            if n == nargs:
                raise Exception('Unmatched parameter name - value pair')

    
    if len(im)==0:
        raise Exception('No image argument supplied')
    
    if not True :
        im = double(im)
    
    if nscale < 1:
        raise Exception('nscale must be an integer >= 1')
    
    if minWaveLength < 2:
        raise Exception('It makes little sense to have a wavelength < 2')
    
    if polarity != - 1 and polarity != 0 and polarity != 1:
        raise Exception('Allowed polarity values are -1, 0 and 1')
    
    #-------------------------------------------------------------------------
# RAYLEIGHMODE
    
    # Computes mode of a vector/matrix of data that is assumed to come from a
# Rayleigh distribution.
    
    # Usage:  rmode = rayleighmode(data, nbins)
    
    # Arguments:  data  - data assumed to come from a Rayleigh distribution
#             nbins - Optional number of bins to use when forming histogram
#                     of the data to determine the mode.
    
    # Mode is computed by forming a histogram of the data over 50 bins and then
# finding the maximum value in the histogram.  Mean and standard deviation
# can then be calculated from the mode as they are related by fixed
# constants.
    
    # mean = mode * sqrt(pi/2)
# std dev = mode * sqrt((4-pi)/2)
    
    # See
# http://mathworld.wolfram.com/RayleighDistribution.html
# http://en.wikipedia.org/wiki/Rayleigh_distribution'''
    
    

""" 
import cv2
import os
from dehazefun import dehazefun 
os.chdir("I:\Drive'ım\matlab codes needing to be py\ImagePhaseCongruency.jl-master\Python")
im = cv2.imread('img.png')



ft, symmetryEnergy = phaseasymmono(im,2,25,3,4,'ASSD',2,.2,3,1,0)


im1 = symmetryEnergy - np.min(symmetryEnergy)
im1 = 255*(im1/np.max(np.max(im1)))
 """

