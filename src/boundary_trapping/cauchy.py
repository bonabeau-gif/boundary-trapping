"""Ordinary real-line Cauchy family under Lorentzian directional fitness."""
import numpy as np

def cauchy_density(x,mu,sigma):
    if sigma<=0:raise ValueError("sigma must be positive")
    return sigma/(np.pi*((np.asarray(x)-mu)**2+sigma*sigma))

def lorentzian_expected_fitness(mu,sigma):
    """E[1/(1+X^2)] for X distributed Cauchy(mu,sigma)."""
    return (1+sigma)/((1+sigma)**2+mu**2)

def lorentzian_endpoint(mu,sigma):
    """Wrong limiting location for Cauchy Fisher projection, sigma>0."""
    if sigma<=0:raise ValueError("sigma must be positive")
    # Streamlines are circles centered at (c,0): c=(mu^2+sigma^2+1)/(2mu)
    if mu==0:return 0.
    c=(mu*mu+sigma*sigma+1)/(2*mu)
    return c-np.sign(c)*np.sqrt(c*c-1)
