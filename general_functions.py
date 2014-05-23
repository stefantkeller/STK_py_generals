#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np

def create_geometric_series( start, end, n ):
    start, end, n = float(start), float(end), float(n) # sanitize input
    if n > 1:
        series = [ start * ( (end/start) ** (float(j)/(n-1)) ) for j in range(n) ]
    else:
        series = [start, end]
    return series
    
def permute_list(l, copy=False):
    """
    if copy==True the input list is not changed.
    The input and returned list have a separate id and occupy thus different pointers. 
    Check with hex(id(l)).
    """
    if isinstance(l, list):
        if copy: l = list(l) # work only with copy of input (hence we can use the same name...)
        for j in range(len(list)):
            k = np.random.randint(j, len(l)-1)
            l[j], l[k] = l[k], l[j]
        return l
    else: # don't do anything
        return l
    
def moving_average(array, order=4, future=False):
    """
    This moving / floating average is good to flatten out noisy data:
    While the actual signal is reproduced by conducting a measurement
    multiple times, the noise cancels (mostly) out. Consequently the
    signal to noise ratio improves as sqrt(n) with n the number of 
    repetitions.
    The moving average works similarly:
    In an array with data points collected in a short time steps
    or time independent but subsequent
        array = [s(t_1), s(t_2), ..., s(t_n)],
    the individual points
        subarray_k = [s(t_k-m/2), ..., s(t_k+m/2)]
    correspond to about the same point s(t_k) (for m reasonably small).
    The average of these points would therefore 'improve' the
    measured point s(t_k):
        s'(t_k) = mean( subarray_k ).
    
    ---------------------------
    parameters
        array: list, as stated above
        order: int, m of subarray
        future: boolean, whether or not to consider future data points
                this affects the subarray_k:
                future is True:
                    subarray_k = [s(t_k-m/2), ..., s(t_k+m/2)]
                future is False:
                    subarray_k = [s(t_k-m), ..., s(t_k)]
                    the average is retarded by tau=(n-1)/2
    returns
        array' = [s'(t_1), s'(t_2), ..., s'(t_n)]
    """
    order = int(order) # to ensure proper behavior: if order is odd, this is thrown away
    if future:
        avgd = [np.mean( array[ max(0, k-order/2 ) : min( k+1+order/2, len(array) ) ] ) for k in range(len(array))]
    else:
        avgd = [np.mean( array[ max(0, k-order ) : min( k+1, len(array) ) ] ) for k in range(len(array))] # in second half up to k+1 and not k+1+order, b/c no future!
    return avgd

def linreg(x,y):
    """
    y = A+Bx
    """
    n = len(x)

    denum = (n*np.sum(x**2) - (np.sum(x))**2)

    Anom = (np.sum(x**2)*np.sum(y) - np.sum(x)*np.sum(x*y))
    Bnom = (n*np.sum(x*y) - np.sum(x)*np.sum(y))

    A = Anom/denum
    B = Bnom/denum

    return A, B


def secant(f,x0,x1,n):
    """
    f to be (e.g. lambda) function
    returns x_ for which f(x_)==0
    terminates after at most n trys
    
    f(x1) = f(x0) + (x1-x0)f'(x0)
    f'(x0) = ( f(x1)-f(x0) )/( x1-x0 )
    f(x0) ~= 0
    => x0 = x1 - f(x1)/f'(x0)
          = x1 - f(x1) * (x1-x0)/(f(x1)-f(x0))
    """
    for i in xrange(n):
        if f(x1)-f(x0) == 0:
            return x1
        tmp = x1 - (f(x1)*(x1-x0)*1.0)/(f(x1)-f(x0))
        x0 = x1
        x1 = tmp
    return x1

def dydx(y,dx):
    # careful: if you don't actually need a function but only
    # an array corresponding to a derivative, then use just
    # (y[:-1]-y[1:])/dx !
    # otherwise this function will potentially take forever
    # to execute!
    return lambda i: ((y[:-1]-y[1:])/dx)[i]

def scaled(data, mode=None):
    data = np.array(data) # "list - n" throws error: we cannot calculate with lists; use np!
    if mode=='01':
        mx, mn = np.max(data), np.min(data)
        data = (data-mn)/(mx-mn)
    elif mode=='+-1':
        data = 2.0*scaled(data,mode='01') - 1
    return data

def find_index(array,value,low=None,high=None):
    '''
    in an array find the index whose corresponding value matches best with requested value
    returns first best match within given interval [low, high]
    '''
    if low==None: low=0
    if high==None: high=len(array)-1

    f = lambda x: array[int(x)]-value
    index = secant(f,low,high,10)
    return int(index)

# ===============================================================
def main():
    pass

if __name__ == '__main__': main()
