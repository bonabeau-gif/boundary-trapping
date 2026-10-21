"""Watson-type symmetric closure under cos² selection on the circle.

q_k(theta) = exp(k*cos(2theta)) / (2*pi*I0(k)).
The symmetric order parameter is E[cos(2theta)] = I1(k)/I0(k).
"""
import numpy as np
from scipy.special import ive
from scipy.integrate import quad

def order_parameter(kappa):
    return float(ive(1,kappa)/ive(0,kappa))

def variance_cos2(kappa):
    # Var(cos 2theta) = dm/dkappa. Stable through Bessel ratios.
    m=order_parameter(kappa)
    if abs(kappa)<1e-8:return .5
    return float(1-m*m-m/kappa)

def fisher_watson_velocity(kappa,rate=0.,mode='diffusion'):
    """Fisher-projected kappa' under cos² selection + mutation.

    Mutation is angular diffusion (rate D) or uniform resetting (rate gamma).
    Fitness cos²theta=(1+cos2theta)/2.
    """
    var=variance_cos2(kappa)
    if mode=='diffusion':
        # m' = .5 Var(cos2theta) - 4D*m
        return .5-4*rate*order_parameter(kappa)/var
    if mode=='reset':
        return .5-rate*order_parameter(kappa)/var
    raise ValueError('unknown mutation mode')
