#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np

"""
github.com/stefantkeller

Convenience class to work with error (aka uncertainty) attached

It is
c = f(a+-da,b+-db)
=> dc = sqrt( (df/da)**2 * da**2 + (df/db)**2 * db**2 )

each implemented operation + - * / ** returns a new instance.
"""

class errval(object):
    def __init__(self,val,err,printout='latex'):
        '''
        val = the value
        err = the corresponding error
        printout = what 'print' should look like
        '''
        self.__val = val
        if err<0: raise ValueError, 'Cannot assign negative error'
        else: self.__err = err
        self.__printout = printout
    
    def val(self):
        return self.__val
    def err(self):
        return self.__err

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
        else: # default = latex
            return "{0} \pm {1}".format(self.__val,self.__err)

    
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
            raise TypeError, 'unsupported operand type(s) for -: errval with {0}'.format(type(other))
        return errval(nval, nerr, self.__printout)
    def __rpow__(self,other):
        if isinstance(other,(int,float,long)):
            # a value with zero error attached
            nval = other ** self.val()
            nerr = abs( np.log(other) * other**self.val() * self.err() )
        else:
            raise TypeError, 'unsupported operand type(s) for -: errval with {0}'.format(type(other))
        return errval(nval, nerr, self.__printout)
    
    def __abs__(self):
        return errval(abs(self.val()), self.err())
    
    
    
    
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
    print abs(c)


if __name__ == '__main__': main()
