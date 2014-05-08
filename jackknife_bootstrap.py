#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

"""
A toy implementation of the bootstrap and jackknife for statistical analysis.
Motivated by lecture on "Statistical Methods and Analysis Techniques in Experimental Physics"
spring semester 2014

Example from
B. Efron, G. Gong, A Leisurely Look at the Bootstrap, the Jachknife and Cross-Validation,
The American Statistician, Vol 37, No 1 (Feb 1983), pp. 36-48
"""


import matplotlib.pyplot as plt
import numpy as np


def rho_xy(data):
    # data = [(x1,y1),(x2,y2),...,(xN,yN)]
    N = len(data)
    x, y = zip(*data)

    x_, y_ = np.mean(x), np.mean(y)
    sigma_x, sigma_y = np.std(x,ddof=1), np.std(y,ddof=1) # denom = 1/(N-ddof)
    
    s_xy = 1.0/(N-1) * np.sum( [(x[i] - x_)*(y[i] - y_) for i in range(N)] )

    r_xy = s_xy/(sigma_x*sigma_y)

    return r_xy

def jackknife(data,foo, output=1):
    # data = [data1,data2,...,dataN] (such that foo can handle it...)
    # foo = function for which you want the error estimate
    # output = how many of the following results should be returned? (if invalid value, the first is returned:)
    #          {sigma, xi (array (length len(data)) result of input when ith element excluded), xdot (mean of xi)}
    #          (if invalid value specified, the first is returned)
    xi = []
    n = len(data)
    for i in range(n):
        data_ = [data[j] for j in range(len(data)) if j!=i]
        xii = foo(data_)
        xi.append(xii)

    xdot = np.mean(xi)

    sigma_J = (len(data)-1)**0.5*np.std([xii-xdot for xii in xi], ddof=0)

    if output==3:
        return sigma_J, xi, xdot
    elif output==2:
        return sigma_J, xi
    else:
        return sigma_J

def bootstrap(data, foo, B, output=1):
    # data = [data1,data2,...,dataN] (such that foo can handle it...)
    # foo = function for which you want the error estimate
    # B = int number of bootstrap repetitions
    # output = how many of the following results should be returned?
    #          {sigma, array (length B) of starred calculations}
    #          (if invalid value specified, the first is returned)
    n = len(data)
    F_star = []
    for b in range(B):
        X_star = [ data[np.random.randint(n)] for i in range(n) ] # "make n random draws with replacement from {x1,x2,...,xn}"
        F_star.append(foo(X_star))

    sigma_B = np.std(F_star,ddof=1) # denom = 1/(N-ddof)

    if output == 2:
        return sigma_B, F_star
    else:
        return sigma_B

def main():
    B = 1000
    LSAT = [576,635,558,578,666,580,555,661,651,605,653,575,545,572,594]
    GPA  = [3.39,3.30,2.81,3.03,3.44,3.07,3.00,3.43,3.36,3.13,3.12,2.74,2.76,2.88,2.96]
    data = zip(LSAT,GPA)
    n = len(data)

    rho0 = rho_xy(data)
    print rho0, '~=',0.776, 'as proposed by Efron 1983:', (abs(0.776-rho0)<0.001)

    # rho
    sigma_J = jackknife(data,rho_xy)
    sigma_B, rho_star = bootstrap(data, rho_xy, B, output=2)
    sigma_rho = lambda rho,n: ((1-rho)*(1+rho))/(n-3)**0.5
    sigma_norm = sigma_rho(rho0,n)

    # mean -> std
    # Efron uses standard error (instead std)
    # sigma_efron = { 1/n(n-1) sum[(xi-<x>)**2] }**1/2 = sigma/sqrt(n)
    efron_correction = np.sqrt( len(data) )
    foo = np.mean
    sigma_x_J = jackknife(LSAT,foo) * efron_correction
    sigma_x_B = bootstrap(LSAT,foo,B) * efron_correction
    sigma_x_  = np.std(LSAT,ddof=1)
    sigma_y_J = jackknife(GPA,foo) * efron_correction
    sigma_y_B = bootstrap(GPA,foo,B) * efron_correction
    sigma_y_  = np.std(GPA,ddof=1)

    print ''
    print 'rho estimation:'
    print 'sigma_J =', sigma_J
    print 'sigma_B =', sigma_B
    print 'sigma_norm =', sigma_norm
    print ''
    print 'ordinary std for x:'
    print 'sigma_x_J =', sigma_x_J
    print 'sigma_x_B =', sigma_x_B
    print 'sigma_x_ =', sigma_x_
    print ''
    print 'ordinary std for y:'
    print 'sigma_y_J =', sigma_y_J
    print 'sigma_y_B =', sigma_y_B
    print 'sigma_y_ =', sigma_y_

    if True:
        plt.subplot(121)
        plt.scatter(LSAT,GPA)
        plt.title('Figure 1')
        plt.xlabel('LSAT')
        plt.ylabel('GPA')

        plt.subplot(122)
        histrange = (-0.5,0.3)
        binres = 0.02
        nbins = (histrange[1]-histrange[0])*1.0/binres
        deltarho = np.array([rs-rho0 for rs in rho_star])
        plt.hist(deltarho,bins=nbins,range=histrange,normed=True)
        plt.title('Figure 2')
        plt.xlabel(r'$\hat{\rho}^*-\hat{\rho}$')
        plt.ylabel('')

        plt.show()


if __name__ == "__main__":
    main()
