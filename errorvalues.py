#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from general_functions import find_index, find_index_iter

"""
https://github.com/stefantkeller/STK_py_generals

Convenience class to work with error (aka uncertainty) attached

It is
c = f(a+-da,b+-db)
=> dc = sqrt( (df/da)**2 * da**2 + (df/db)**2 * db**2 )

each implemented operation {+, -, *, /, **, abs()} returns a new instance.
"""

class errval(object):
    def __init__(self,val,err=0,printout='latex'):
        '''
        val = the value (number or errval (see below))
        err = the corresponding error
        printout = what 'print' should look like

        if you initiate with val=errval a copy of that input errval is provided
        if you then want to change the printout value for that copy you have
        two choices:
            (1) a posteriori with .printout('mynewopt')
            (2) by specifying the new option with a !bang:
                cp = errval(orig,printout='mynewopt!')
        '''
        if isinstance(val,errval): # return copy
            self.__val = val.val()
            self.__assign_err(val.err())
            self.__printout = val.printout()
            if printout.endswith('!'): self.__printout = printout
        elif ( isinstance(val,tuple) and len(val)==2
               and isinstance(val[0],(int,float,long)) and isinstance(val[1],(int,float,long)) ):
            self.__val = val[0]
            self.__assign_err(val[1])
            self.__printout = printout
        elif ( isinstance(val,tuple) and len(val)==3
               and isinstance(val[0],(int,float,long)) and isinstance(val[1],(int,float,long))
               and isinstance(val[2],str) ):
            self.__val = val[0]
            self.__assign_err(val[1])
            self.__printout = val[2]
        elif isinstance(val,(int,float,long)):
            self.__val = val
            self.__assign_err(err)
            self.__printout = printout
        else:
            raise ValueError, 'Cannot assign input data'

    
    def val(self):
        return self.__val
    def err(self):
        return self.__err
    def v(self):
        return self.__val
    def e(self):
        return self.__err
    def printout(self,change=''):
        if change!='':
            self.__printout = change
        return self.__printout


    def __str__(self):
        '''
        called by print
        output depends on initialization
        when two errvals are put together the result has the value of the left one
        '''
        if self.__printout == '+-':
            return "{0} +- {1}".format(self.__val,self.__err)
        if self.__printout == 'cp': # make it easier to copy paste...
            return "errval({0},{1})".format(self.val(),self.err())
        if self.__printout == 'cpp': # make it easier to copy paste...
            return "errval({0},{1},errvalmode)".format(self.val(),self.err())
        else: # default = latex
            return "{0} \pm {1}".format(self.__val,self.__err)

    def __assign_err(self,err):
        if err<0: raise ValueError, 'Cannot assign negative error'
        else: self.__err = err
    
    def __add__(self,other):
        if isinstance(other,errval):
            nval = self.val() + other.val()
            nerr = np.sqrt( self.err()**2 + other.err()**2 )
        elif isinstance(other,(int,float,long)):
            # a value with zero error attached
            nval = self.val() + other
            nerr = self.err()
        else:
            raise TypeError, 'unsupported operand type(s) for +: errval with {0}'.format(type(other))
        return errval(nval, nerr, self.__printout)
    def __radd__(self,other):
        return self.__add__(other)
    
    def __sub__(self,other):
        if isinstance(other,errval):
            nval = self.val() - other.val()
            nerr = np.sqrt( self.err()**2 + other.err()**2 )
        elif isinstance(other,(int,float,long)):
            # a value with zero error attached
            nval = self.val() - other
            nerr = self.err()
        else:
            raise TypeError, 'unsupported operand type(s) for -: errval with {0}'.format(type(other))
        return errval(nval, nerr, self.__printout)
    def __rsub__(self,other):
        if isinstance(other,(int,float,long)):
            # a value with zero error attached
            nval = other - self.val()
            nerr = self.err()
        else:
            raise TypeError, 'unsupported operand type(s) for -: {0} with errval'.format(type(other))
        return errval(nval, nerr, self.__printout)
        
    def __mul__(self,other):
        if isinstance(other,errval):
            nval = self.val() * other.val()
            nerr = np.sqrt( (other.val()*self.err())**2 + (self.val()*other.err())**2 )
        elif isinstance(other,(int,float,long)):
            # a value with zero error attached
            nval = self.val() * other
            nerr = self.err() * abs(other)
        else:
            raise TypeError, 'unsupported operand type(s) for *: errval with {0}'.format(type(other))
        return errval(nval, nerr, self.__printout)
    def __rmul__(self,other):
        return self.__mul__(other)
    
    def __div__(self,other):
        if isinstance(other,errval):
            nval = self.val() *1.0 / other.val()
            nerr = np.sqrt( (1.0/other.val()*self.err())**2 + (self.val()*1.0/(other.val()**2)*other.err())**2 )
        elif isinstance(other,(int,float,long)):
            # a value with zero error attached
            nval = self.val() *1.0 / other
            nerr = self.err() *1.0/ abs(other)
        else:
            raise TypeError, 'unsupported operand type(s) for /: errval with {0}'.format(type(other))
        return errval(nval, nerr, self.__printout)
    def __rdiv__(self,other):
        if isinstance(other,(int,float,long)):
            # a value with zero error attached
            nval = other *1.0/self.val()
            nerr = abs(other)*1.0/self.val()**2 * self.err()
        else:
            raise TypeError, 'unsupported operand type(s) for /: {0} with errval'.format(type(other))
        return errval(nval, nerr, self.__printout)
    
    def __pow__(self,other):
        if isinstance(other,errval):
            nval = self.val() ** other.val()
            nerr = np.sqrt( ( other.val() * self.val()**(other.val()-1) * self.err() )**2
                            + ( np.log(self.val()) * self.val()**other.val() * other.err() )**2 )
        elif isinstance(other,(int,float,long)):
            # a value with zero error attached
            nval = self.val() ** other
            nerr = abs( other * self.val()**(other-1) * self.err() )
        else:
            raise TypeError, 'unsupported operand type(s) for **: errval with {0}'.format(type(other))
        return errval(nval, nerr, self.__printout)
    def __rpow__(self,other):
        if isinstance(other,(int,float,long)):
            # a value with zero error attached
            nval = other ** self.val()
            nerr = abs( np.log(other) * other**self.val() * self.err() )
        else:
            raise TypeError, 'unsupported operand type(s) for **: errval with {0}'.format(type(other))
        return errval(nval, nerr, self.__printout)
    
    def __abs__(self):
        return errval(abs(self.val()), self.err(), self.printout())
    
    def sqrt(self):
        return self**0.5

    def round(self,n=0):
        # returns new instance
        return errval(np.around(self.val(),n),np.around(self.err(),n),self.printout())


def stderrval(li):
    # errval with standard error from a list
    # standard error being the unbiased std divided by n
    return errval(np.mean(li),\
                   1.0/np.sqrt(len(li))*np.std(li,ddof=1))

#---------------------------------------------------------------------------------------

class errvallist(list):
    def __init__(self,vals,errs=0,printout='latex'):
        if isinstance(vals,errvallist):
            self.__errl = vals
        elif isinstance(vals,(list,np.ndarray)) or isinstance(errs,(list,np.ndarray)):
            if isinstance(vals,(list,np.ndarray)) and isinstance(errs,(list,np.ndarray)):
                if len(vals)==len(errs):
                    self.__errl = [errval(vals[j],errs[j],printout) for j in xrange(len(vals))]
            elif isinstance(vals,(list,np.ndarray)) and isinstance(errs,(int,float,long)):
                # note: this also covers the case vals is a list of errval entries,
                # in this case the errs are ignored and the result is a conversion of list to errvallist
                self.__errl = [errval(v,errs,printout) for v in vals]
            elif isinstance(vals,(int,float,long)) and isinstance(errs,(list,np.ndarray)):
                self.__errl = [errval(vals,e,printout) for e in errs]
        else:
            raise ValueError, 'Cannot assign input data: {0}'.format(type(vals))

    def __getitem__(self,key):
        return self.__errl[key]
    def __setitem__(self,key,value):
        self.__errl[key]=value

    def __str__(self):
        outp = '['
        for evl in self.__errl:
            outp += evl.__str__()+','
        outp = outp[:-1]+']'
        return outp
    
    def __iter__(self):
        # to make the errvallist iterable
        # i.e. to make the 'in' possible in 'for err in errvallist:'
        for err in self.__errl:
            yield err

    def __len__(self):
        return len(self.__errl)

    def __add__(self,other):
        if isinstance(other,(errvallist,list)) and len(self)==len(other):
             errvall = [self[j]+other[j] for j in xrange(len(self))]
        elif isinstance(other,(int,float,long,errval)):
            errvall = [s+other for s in self]
        else:
            raise TypeError, 'unsupported operand type(s) for +: errval with {0}'.format(type(other))
        return errvallist(errvall)
    def __radd__(self,other):
            return self.__add__(other)

    def __sub__(self,other):
        if isinstance(other,(errvallist,list)) and len(self)==len(other):
            errvall = [self[j]-other[j] for j in xrange(len(self))]
        elif isinstance(other,(int,float,long,errval)):
            errvall = [s-other for s in self]
        else:
            raise TypeError, 'unsupported operand type(s) for -: errval with {0}'.format(type(other))
        return errvallist(errvall)
    def __rsub__(self,other):
        return -1*self.__sub__(other)

    def __mul__(self,other):
        if isinstance(other,(errvallist,list)) and len(self)==len(other):
            errvall = [self[j]*other[j] for j in xrange(len(self))]
        elif isinstance(other,(int,float,long,errval)):
            errvall = [s*other for s in self]
        else:
            raise TypeError, 'unsupported operand type(s) for *: errval with {0}'.format(type(other))
        return errvallist(errvall)
    def __rmul__(self,other):
        return self.__mul__(other)

    def __div__(self,other):
        if isinstance(other,(errvallist,list)) and len(self)==len(other):
            errvall = [self[j]/other[j] for j in xrange(len(self))]
        elif isinstance(other,(int,float,long,errval)):
            errvall = [s/other for s in self]
        else:
            raise TypeError, 'unsupported operand type(s) for /: errval with {0}'.format(type(other))
        return errvallist(errvall)
    def __rdiv__(self,other):
        return 1.0/self.__div__(other)

    def append(self,value):
        self.__errl.append(value)

    '''
    Depending on the circumstances the code incorporating this class
    may want to use different names for the following functions:
    '''
    def v(self): return values(self)
    def val(self): return self.v()
    def vals(self): return self.v()
    def values(self): return self.v()

    def e(self): return errors(self)
    def err(self): return self.e()
    def errs(self): return self.e()
    def errors(self): return self.e()


def stderrvallist(li):
    # errvallist with standard errors from a n-dim list
    # standard error being the unbiased std divided by n
    # input li = [[1,2,3],
    #             [2,3,4]]
    # outp res = [1.5+-0.5,2.5+-0.5,3.5+-0.5]
    return errvallist(np.mean(li,axis=0),\
                       1.0/np.sqrt(len(li))*np.std(li,axis=0,ddof=1))

'''
functions for dealing with lists containing errvals
not necessarily errvallists
'''
def values(errvallist):
    return np.array([ev.val() for ev in errvallist])
def errors(errvallist):
    return np.array([ev.err() for ev in errvallist])
def tuples(errvallist):
    return zip(values(errvallist),errors(errvallist))

def find_fooval(errvallist,foo,index=False):
    '''
    Find value in list closes to foo(list)
    e.g. foo=max returns the maximum value in the list
    caution: this might not terminate
    (unclear why, open question over at general_functions)
    if not just 'closet' but exact value looked for,
    pick find_fooval_iter
    '''
    v = values(errvallist)
    i = find_index(v,foo(v))
    if index: return errvallist[i], i
    else: return errvallist[i]
def find_fooval_iter(errvallist,foo,index=False):
    '''
    Find value in list corresponding exactly to foo(list)
    e.g. foo=max returns the maximum value in the list
    '''
    v = values(errvallist)
    i = find_index_iter(v,foo(v))
    if index: return errvallist[i], i
    else: return errvallist[i]

def max_(errvallist,index=True):
    # there is only one value,
    # and that one is exact, so go with _iter:
    return find_fooval_iter(errvallist,max,index)
def min_(errvallist,index=True):
    return find_fooval_iter(errvallist,min,index)

def wmean(errvallist):
    '''
    weighted mean

    sigma_<x>^2 = sum(1/sigma_i^2)
    <x> = sum(x_i/sigma_i^2)/sigma_<x>^2
    '''
    printmode = errvallist[0].printout()
    vals = values(errvallist)
    #print vals
    errs = errors(errvallist)
    #print errs
    N = len(errvallist)
    sig_x = np.sum([1.0/si**2 for si in errs])
    sum_x = np.sum([vals[i]*1.0/errs[i]**2 for i in range(N)])
    return errval(sum_x*1.0/sig_x,1.0/np.sqrt(sig_x),printmode)
    
def interp(v,evxy0,evxy1):
    '''
    linear interpolation between two points
    evxy0 (evxy1) is expected to be a tuple
    representing the x and y value of the
    point to the left (right) of value v
    otherwise it's an extrapolation - proceed with caution!

    Basic idea (for values between v \in [0,1]):
        y = y0 + v*(y1-y0)
    But v is not bound to [0,1], so we have to renormalize:
        v' = (v-x0)/(x1-x0).
    This makes it implicitly clear that we expect x1>x0,
    so, order your input properly!
    We end up:
        y = y0 + (v-x0)/(x1-x0)*(y1-y0).
    '''
    if not isinstance(evxy0,tuple) or not isinstance(evxy1,tuple):
        raise TypeError,\
             'Boundary required as tuple, {0} and {1} given'.format(
                type(evxy0),type(evxy1))
    # if input can be handled with pre-existing functions:
    if not isinstance(evxy0[0],errval) and \
        not isinstance(evxy0[1],errval) and \
        not isinstance(evxy1[0],errval) and \
        not isinstance(evxy1[1],errval):
        return np.interp(v,evxy0,evxy1)
    # ok, at least one of the inputs is errval; worth the time:
    # make sure every entry is indeed errval
    # (because lazy, the x-values don't need to have an error,
    # in fact, if they do this error will be lost)
    x0, y0 = errval(evxy0[0]), errval(evxy0[1])
    x1, y1 = errval(evxy1[0]), errval(evxy1[1])

    scaling = float(v-x0.v())/(x1.v()-x0.v())
    y = y0.v() + scaling*(y1.v()-y0.v())
    ye = y0.e() + scaling*abs(y1.e()-y0.e())
    #scalingy = (y-y0.v())/(y1.v()-y0.v())
    #xe = scalingy*abs(x1.e()-x0.e())
    return errval(y,ye)

def interplist(v,evx,evy):
    '''
    evx must be an ordered (errval,)list or a numpy.ndarray
    evy must be a errvallist
    '''
    if not isinstance(evx,(list,tuple,errvallist,np.ndarray)):
        raise TypeError, 'evx is of unexpected type: {0}'.format(
                type(evx))
    if not isinstance(evy,errvallist):
        raise TypeError, 'This function is for errvallists, try np.interp'
    if isinstance(evx,errvallist):
        evx = evx.v() # errval has only one-dimensional error
    i0 = np.sum([e<=v for e in evx])
    if i0==0: raise ValueError,\
                'Value below interpolation values: {0}<{1}'.format(
                    i0,evx[0])
    xy0 = (evx[i0-1],evy[i0-1])
    xy1 = (evx[i0],evy[i0])
    return interp(v,xy0,xy1)
# ---------------------------------------------------------------------------------------
    
def main():
#    z = errval(0,-1)
    aa, da = 1, 3
    a = errval(aa, da)
    bb, db = 2, 4
    b = errval(bb, db, '+-')
    cc, dc = -3, 5
    c = errval(cc,dc)
    # test basic output functionality
    print 'exp: {0} {1} \ngot: {2} {3}\n---'.format(aa,da, a.val(), a.err())
    print 'exp: {0} \pm {1} \ngot: {2}\n---'.format(aa,da,a)
    print 'exp: {0} +- {1} \ngot: {2}\n---'.format(bb,db,b)
    print 'exp: <class \'__main__.errval\'> True \ngot: {0} {1}\n---'.format(type(b), isinstance(b,errval))
    
    
    # some arithmetic
    d0, d1, d2 = a+b, b+a, a+2
#    d0+[] # should raise error
    e0, e1, e2 = a-b, a-2, 1-b
#    e0-[] # should raise error
    f0, f1, f2, f3 = a+b+c, a-b+c, a+b-c, b+a-c
    print 'exp: {0} \pm {1} \ngot: {2}\n---'.format(3,5,d0)
    print 'exp: {0} +- {1} \ngot: {2}\n---'.format(3,5,d1)
    print 'exp: {0} \pm {1} \ngot: {2}\n---'.format(3,3,d2)
    print 'exp: {0} \pm {1} \ngot: {2}\n---'.format(-1,5,e0)
    print 'exp: {0} \pm {1} \ngot: {2}\n---'.format(-1,3,e1)
    print 'exp: {0} +- {1} \ngot: {2}\n---'.format(-1,4,e2)
    print 'exp: {0} \pm {1} \ngot: {2}\n---'.format(0,np.sqrt(50),f0)
    print 'exp: {0} \pm {1} \ngot: {2}\n---'.format(-4,np.sqrt(50),f1)
    print 'exp: {0} \pm {1} \ngot: {2}\n---'.format(6,np.sqrt(50),f2)
    print 'exp: {0} +- {1} \ngot: {2}\n---'.format(6,np.sqrt(50),f3)
    
    g0, g1, g2, g3, g4 = a*b, b*a, a*2, 2*b, a*b*c
    print 'exp: {0} \pm {1} \ngot: {2}\n---'.format(2,np.sqrt(52),g0)
    print 'exp: {0} +- {1} \ngot: {2}\n---'.format(2,np.sqrt(52),g1)
    print 'exp: {0} \pm {1} \ngot: {2}\n---'.format(2,6,g2)
    print 'exp: {0} +- {1} \ngot: {2}\n---'.format(4,8,g3)
    print 'exp: {0} \pm {1} \ngot: {2}\n---'.format(-6,np.sqrt(568),g4)
    
    h0, h1, h2 = a/b, b/2.0, 1.0/c
    print 'exp: {0} \pm {1} \ngot: {2}\n---'.format(0.5,np.sqrt(3.25),h0)
    print 'exp: {0} +- {1} \ngot: {2}\n---'.format(1,2,h1)
    print 'exp: {0} \pm {1} \ngot: {2}\n---'.format(1.0/-3,5.0/9,h2)
    
    k0, k1, k2 = a**2, 2**a, b**a
    print 'exp: {0} \pm {1} \ngot: {2}\n---'.format(1,6,k0)
    print 'exp: {0} \pm {1} \ngot: {2}\n---'.format(2,np.log(2)*2*3,k1)
    print 'exp: {0} +- {1} \ngot: {2}\n---'.format(2,np.sqrt(16+(np.log(2)*2*3)**2),k2)

    m0 = errval(b,printout='latex!')
    print 'exp: {0} != {1}\ngot: {2}\n---'.format(id(m0),id(b),id(m0)!=id(b))
    print 'exp: {0} \pm {1}\ngot: {2}\n---'.format(b.val(),b.err(),m0)

    n0 = np.sqrt(m0)
    print 'exp: {0} \pm {1} \ngot: {2}\n---'.format(np.sqrt(2),4/(2*np.sqrt(2)),n0)

    p0, p1 = errval((1,2)), errval((1,2,'+-'))
#    p2 = errval((1,2,3)) # error because input nonsense
#    p3 = errval([]) # also.
    print p0
    print p1
    print '---'
    
    abc = [a,b,c]
    abcv, abce, abct = values(abc), errors(abc), tuples(abc)
    print 'exp: {0}\ngot: {1}\n---'.format(f0,np.sum(abc))
    print abcv
    print abce
    print zip(abcv,abce)
    print abct
    print '---'
    abc_ = errvallist([a,b,c])
    print 'Remember: {0}'.format(abc_)
    print 'exp: {0}\ngot: {1}\n---'.format(values(abc),abc_.v())
    print 'exp: {0}\ngot: {1}\n---'.format(errors(abc),abc_.errs())
    print 'exp: {0}\ngot: {1}\n---'.format(f0,np.sum(abc_))
    print 'exp: {0},{1}\ngot: {2},{3}\n---'.format(c,2,min_(abc_)[0],min_(abc_)[1])

    q0, q1 = errval(100,10), errval(1,1)
    r0, r1 = errval(3.11,0.02), errval(3.13,0.01)
    print 'exp: {0}\ngot: {1}\n---'.format(errval(2,1),wmean([q0,q1]))
    print 'exp: {0}\ngot: {1}\n---'.format(errval(3.126,0.009),wmean([r0,r1])) # [R. Barlow, Statistics, John Wiley & Sons Ltd. (1989)]
    print 'exp: {0}\ngot: {1}\n---'.format(errval(3.126,0.009),wmean(errvallist([r0,r1])))
    print 'exp: {0}\ngot: {1}\n---'.format(errval(50.5,0.5*np.sqrt(101)),np.mean(errvallist([q0,q1])))

    print 'Interpolate:'
    s0,t0 = 1,2
    try: print interp(1.5,s0,t0)
    except TypeError: print 'Non tuple input caught. (Good!)'
    s1,t1 = (1,2),(2,2)
    print 'exp: {0}\ngot: {1}\n---'.format(2,interp(1.5,s1,t1))
    s2,t2 = (1,errval(2,1)),(2,errval(1,0))
    print 'exp: {0}\ngot: {1}\n---'.format(errval(1.5,0.5),interp(1.5,s2,t2))
    s3,t3 = (1,errval(3,1)),(3,(errval(1,3)))
    t3_ = interp(2,s3,t3)
    print 'exp: {0}\ngot: {1}\n---'.format(errval(2,2),t3_)
    s4,t4 = [1,3], errvallist([errval(3,1),errval(1,3)])
    t4_ = interplist(2,s4,t4)
    print 'exp: {0}\ngot: {1}\n---'.format(errval(2,2),t4_)
    
    print 'exp: {0}\ngot: {1}\n---'.format(errval(0.5,0.5),
                                            stderrval([0,1]))
    u = [[1,2,3],[2,3,4]]
    u0 = stderrvallist(u)
    v = errvallist([1.5,2.5,3.5],[0.5,0.5,0.5])
    print 'exp: {0}\ngot: {1}\n---'.format(v,u0)
    


if __name__ == '__main__': main()
