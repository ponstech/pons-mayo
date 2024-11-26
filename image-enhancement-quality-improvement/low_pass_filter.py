
from filtergrid import filtergrid

def lowpassfilter(sze, cutoff, n):

    assert cutoff > 0 and cutoff < 0.5, ("cutoff frequency must be between 0 and 0.5")
    assert n >= 0, ("n must be an integer >= 1")

    f,_,__ = filtergrid(sze[0],sze[1])
    
    return 1.0 / (1.0 + (f / cutoff)**(2*n))



