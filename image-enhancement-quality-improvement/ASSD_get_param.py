# from scipy.special import lambertw
import warnings
# import scipy.optimize
import numpy as np


def fmin(func, x0, args=(), maxiter=None, maxfun=None, xtol=1e-4, ftol=1e-4, disp=False):
    """
    Minimize a function using the Nelder-Mead algorithm.

    Parameters
    ----------
    func : callable
        The objective function to be minimized.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to the objective function.
    maxiter : int, optional
        Maximum number of iterations to perform.
    maxfun : int, optional
        Maximum number of function evaluations to make.
    xtol : float, optional
        Absolute error in xopt between iterations that is acceptable for convergence.
    ftol : float, optional
        Relative error in func(xopt) between iterations that is acceptable for convergence.
    disp : bool, optional
        Set to True to print convergence messages.

    Returns
    -------
    x : ndarray
        The solution.
    fx : float
        The value of the function at the solution.
    """
    if maxiter is None:
        maxiter = len(x0)*200
    if maxfun is None:
        maxfun = maxiter*10

    # Create initial simplex
    n = len(x0)
    sim = np.zeros((n+1, n))
    sim[0] = x0
    for i in range(n):
        x = np.array(x0)
        x[i] += 0.05*x[i]
        sim[i+1] = x

    # Evaluate function at simplex vertices
    fsim = np.zeros((n+1,))
    for i in range(n+1):
        fsim[i] = func(sim[i], *args)

    # Initialize iteration counter
    k = 0

    # Main optimization loop
    while k < maxiter and funcalls < maxfun:
        ind = np.argsort(fsim)
        sim = sim[ind]
        fsim = fsim[ind]

        # Check for convergence
        if 2.0*np.abs(fsim[-1]-fsim[0])/(np.abs(fsim[-1])+np.abs(fsim[0])+1e-20) < ftol and \
           np.max(np.abs(sim[1:]-sim[0])) < xtol:
            break

        # Compute centroid
        x0 = np.sum(sim[:-1], axis=0)/n

        # Reflect
        xr = x0 + (x0 - sim[-1])
        fxr = func(xr, *args)
        funcalls += 1

        if fxr < fsim[0]:
            # Expand
            xe = x0 + 2.0*(xr - x0)
            fxe = func(xe, *args)
            funcalls += 1
            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        else:
            if fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else:
                # Contract
                if fxr < fsim[-1]:
                    xc = x0 + 0.5*(xr - x0)
                    fxc = func(xc, *args)


def lambertw(z, k=1e-8):
    x = z
    tol = k
    
    # Initialize values
    w = np.log(np.abs(x) + 1e-5) - np.log(np.abs(np.log(np.abs(x) + 1e-5)) + 1e-5)
    err = np.inf
    iters = 0
    
    # Loop until convergence
    while err > tol:
        w_new = w - (w*np.exp(w) - x)/(np.exp(w)*(w+1) - (w+2)*(w*np.exp(w) - x)/(2*w+2))
        err = np.abs(w_new - w)
        w = w_new
        iters += 1
        if iters > 100:
            raise ValueError("Lambert W function did not converge.")
    
    return w


def ASSD_get_param(beta = None,a = None,alpha = None): 
    # [beta, a]=ASSD_get_param(beta, a, alpha);
    
    # Alpha space wavelets parameters
# beta = Bd(Octave), 1 <= a or alpha in (0,1]
    
    # This function calculates "beta" if "a" is given
# or calculates "a" if "beta" is given, according to "alpha" parameter
    
    #------------------------------------------------------------------------
# Description: This code implements a part of the paper:
# Ahror BELAID and Djamal BOUKERROUI. "A new generalised alpha scale
# spaces quadrature filters." Pattern Recognition 47.10 (2014): 3209-3224.
    
    # Coded by: Ahror BELAID and Djamal BOUKERROUI
#------------------------------------------------------------------------
    
    # Date of creation: Mars 2013
    
    # Copyright (c), Heudiasyc laboratory, Compi�gne, France.
    
    assert alpha <=  1 and alpha > 0, f"alpha must be in (0,1] but got: {alpha}"
    
    #find a given beta

    #find a given beta
    betaMAX = Alpha_Spacebeta(1, alpha = 0.2)
    #find a given beta
    if beta != None :
        for i in range(beta):
            if beta > betaMAX:
                warnings.warn('No "a" for this bandwith %f',beta(i))
                a[i] = - 1
                continue
            #Option = optimset('fminsearch')
            x0 = 1
#             a = scipy.optimize.fmin(func=Alpha_Spacebeta, x0 = x0)
            a = fmin(func=Alpha_Spacebeta, x0 = x0)                                                            
            
            return a
            #a[i],fval,exitflag = fminsearch(Alpha_Spacebeta,1.5,Option,beta(i))
            '''if exitflag != 1:
                warnings.warn('No "a" for this bandwith %f',beta(i))
                a[i] = - 1'''
    else:
        if np.amin(a) < 1:
            raise Exception('a >=1')
        beta = Alpha_Spacebeta(a)
        return beta
    #######################################################
    
def Alpha_Spacebeta(a,alpha = 0.2): 
    
    c = - 1 / (np.exp(1) * 2 ** (2 * alpha / a))
    beta = np.log(lambertw(k = - 1,z = c) / lambertw(k = 0,z= c)) / (np.log(2) * 2 * alpha)
    
    return np.real(beta)