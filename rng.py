#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

class RNG(object):
    """
    Congruential random number generator,
    from one of the exercises of lecture fall 2013
    http://www.ifb.ethz.ch/education/IntroductionComPhys
    
    Note: 
    -) maximally useful amount is p-1, after that sequence repeats
    -) seed=0 cannot be used (selfexplanatory when you look at next())
    -) according to a 1910 proof by R.D. Carmichael to actually get this
       full p-1 period the numbers p and c must fullfill:
       a) p is Mersenne prime, i.e. p = 2**n - 1
       b) c**(p-1) % p == 1
    
    tl;dr:
       only change the seed, leave c and p as set, i.e. use only as
       > r = RNG(seed)
    """
    def __init__(self,seed=42,c=16807,p=2147483647):
        self._xi=float(seed)
        self._c=float(c)
        self._p=float(p)
    
    def next(self):
        self._xi = (self._c*self._xi) % self._p
        return self._xi
    
    def r(self):
        return self.next()/self._p
    
    def bi(self, threshold=0.5):
        # useful for binary decisions... i.e. coin flips
        return 0 if self.r()>threshold else 1


if __name__ == '__main__': pass