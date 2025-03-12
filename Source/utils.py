import numpy as np

def nll_gev(data, p):
    """
    Usefull to compute the variance-covariance matrix using the hessian from numdifftools
    """
    xi = -p[2]
    mu = p[0]
    sigma = p[1]
    
    # Gumbel 
    if np.abs(xi) < 1e-8:
        expr = (data-mu)/sigma
        return (len(data)*np.log(sigma) + np.sum(expr) + np.sum(np.exp(-expr)))

    # Weibull-Frechet 
    else:
        expr = 1+xi*((data-mu)/sigma)
        return (len(data)*np.log(sigma) + (1+1/xi)*np.sum(np.log(expr)) + np.sum(expr **(-1/xi)))

def dq_gev(prob,p):
    """
    Quantile derivatives 
    """
    xi = -p[2]
    mu = p[0]
    sigma = p[1]

    if np.abs(xi) < 1e-8:
        dmu = 1*np.ones_like(prob)
        dsigma = -np.log(-np.log(prob))
        dxi = np.zeros_like(prob)
    else:
        dmu = 1*np.ones_like(prob)
        dsigma = ((-np.log(prob))**(-xi)-1)/xi
        dxi = -(sigma*(np.log(-np.log(prob))*xi - (-np.log(prob))**xi +1))/(((-np.log(prob))**xi) * xi **2)

    Dq = np.zeros((3,len(prob)))

    Dq[0,:] = dmu
    Dq[1,:] = dsigma
    Dq[2,:] = dxi

    return Dq
    