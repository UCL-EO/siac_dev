import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack.realtransforms import dct,idct
import scipy.sparse


def diffFilter(yshape,order=1,axis=(0,1)):
    '''
    Sqrt DCT-II filter for D2 == D^T D
    
    Returns eigenvalues for order p filter. NB
    Square it when using to obtain the correct filter.
    '''
    ndim = len(yshape)
    Lambda = np.zeros(yshape).astype(float)

    for c,i in enumerate(axis):
        # create a 1 x d array (so e.g. [1,1] for a 2D case
        siz0 = np.ones((1,ndim)).astype(int)[0]
        siz0[i] = yshape[i]
        omega = np.pi*np.arange(yshape[i])/float(yshape[i])
        this = np.cos(omega).reshape(siz0)
        Lambda = Lambda + this
    Lambda = -(len(axis)-Lambda)
    return np.abs((2*Lambda))**(order/2.)

def dctND(data,f=dct,axis=(0,1)):
    for i in np.arange(len(axis)):
        data = f(data,norm='ortho',type=2,axis=axis[i])
    return data

def filt(y,dctFilter,thresh=1e-10,fwd=True,axis=(0,1),do_dct=True):
    '''
    filter y by dctFilter
    with dctFilter in DCT space
    
    if ,do_dct=False, then y is already in dct space
    ''' 
    if fwd:
      dd = [dct,idct]
    else:
      dd = [idct,dct]

    shape = dctFilter.shape
    if do_dct:
        dcty = dctND(y.reshape(shape),f=dd[0],axis=axis)
    else:
        y = dcty
    DTDy = dctND(dctFilter * dcty,f=dd[1],axis=axis)
    DTDy[np.logical_and(DTDy>=-thresh,DTDy<=thresh)] = 0.
    return DTDy

