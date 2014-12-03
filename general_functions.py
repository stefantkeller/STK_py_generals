#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np

from itertools import izip as zip, count # izip for maximum efficiency: http://stackoverflow.com/questions/176918/finding-the-index-of-an-item-given-a-list-containing-it-in-python

def create_geometric_series( start, end, n ):
    start, end, n = float(start), float(end), int(n) # sanitize input
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
        for j in range(len(l)):
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
    works with 
    returns first best match within given interval [low, high]
    '''
    if low==None: low=0
    if high==None: high=len(array)-1

    f = lambda x: array[int(x)]-value
    index = secant(f,low,high,10)
    return int(index)

def find_index_iter(array,value,low=None,high=None):
    '''
    same as find_index() but the value has to be exact
    [low,high] constrains search if approx position in the array is known already

    example:
    a = [1,2,3,4,5,6,7,8,9,0]
    it = find_index_iter(a,4)
    print it, (it==3)
    it = find_index_iter(a,8,high=6)
    print it, (it==None)
    it = find_index_iter(a,8,high=7)
    print it, (it==7)
    it = find_index_iter(a,0,low=7)
    print it, (it==9)
    '''
    if low==None: low=0
    if high==None: high=len(array)-1

    it = low
    for search in xrange(high-low+1):
    # one more iteration than range: to get last iteration as well.
    # if the value is found, this loop should break
    # if it doesn't the last +=1 is executed, which we can detect.
        if array[it]==value: break
        it += 1
    if it==high+1: it=None # not found; loop didn't break
    return it

def llist(instr,outtype=list):
    '''
    returns a long, flat list of type outtype,
    whose entries are of value a with multiplicity b

    example 1:
    instr = [(a1,b1),(a2,b2),...]
    out = [a1,a1,a1,...,a2,a2,...]
            \- b1 -/    \- b2 -/
    
    example 2:
    what_i_want = [(1,2), # 1,1
                   (2,3), # 2,2,2
                   (1,5)] # 1,1,1,1,1
    llist(what_i_want)
    >>> [1,1,2,2,2,1,1,1,1,1]

    example 3:
    in MATLAB it is:
    a = [0*ones(1,250),8*ones(1,1000),6.5*ones(1,1000)]
    with llist it is:
    a = llist([(0,250),(8,1000),(6.5,1000)])
    '''
    out = []
    for i in instr:
        if len(i)==2 and isinstance(i[1],int):
            # note for when bored: benchmark the following
            out += [i[0]]*i[1]
            #out += [i[0] for k in xrange(i[1)]
    return outtype(out)

def split2dict(strng,el_sep=';',en_sep='=',ignore_wrong_entries=True):
    # 'a=1;b=2;c=3' -> {'a': '1', 'c': '3', 'b': '2'}
    # strng = 'a=1;b=2;c=3'
    # el_sep: element ('a=1',...) separator, e.g. ';'
    # en_sep: entry ('a'='1',...) separator, e.g. '='
    # if not ignore_wrong_entries:
    #   string whose parts doesn't fit above described pattern raises an error
    #   e.g. 'a=1;without equal sign;valid=True' raises a ValueError
    # if ignore_wrong_entries:
    #   these ValueErrors are ignored (pass)
    el = strng.split(el_sep)
    splt = lambda x: x.split(en_sep)
    retdict = {} # return dict( (e0,e1) for e0,e1 in map( splt, el ) )
    for e in el:
        try:
            e0,e1 = splt(e)
            retdict[e0]=e1
        except ValueError, e:
            if ignore_wrong_entries: pass
            else:
                print strng
                raise ValueError, e
    return retdict
        

def find_random_duplicates(input):
    # input = [10,11,12,13,14,13,12,10,11,11,12,13,14,10,14]
    # (indices[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14])
    # return: {10: [0, 7, 13], 11: [1, 8, 9], 12: [2, 6, 10], 13: [3, 5, 11], 14: [4, 12, 14]}
    #http://stackoverflow.com/questions/176918/finding-the-index-of-an-item-given-a-list-containing-it-in-python
    #http://stackoverflow.com/questions/479897/how-do-you-remove-duplicates-from-a-list-in-python-if-the-item-order-is-not-impo
    deduplicated = sorted(set(input))
    indexlist = [[i for i, j in zip(count(), input) if j == k] for k in deduplicated]
    return dict((e0,e1) for e0,e1 in zip(deduplicated,indexlist))

# ===============================================================
def main():
    pass

if __name__ == '__main__': main()
