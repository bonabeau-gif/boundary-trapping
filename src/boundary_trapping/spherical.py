"""Dimension-four analytic spherical-Cauchy harmonic factors."""
import numpy as np

def d4_h1(r):return (3*r-r**3)/2

def d4_h2(r):return 2*r*r-r**4

def d4_linear_objective(a):
    a=np.asarray(a,dtype=float)
    r=np.linalg.norm(a)
    if r>=1:raise ValueError("parameter outside unit ball")
    return (d4_h1(r)/r*a[0]) if r>0 else 0.

def d4_linear_gradient(a):
    a=np.asarray(a,dtype=float); r=np.linalg.norm(a)
    e=np.zeros_like(a);e[0]=1
    return (1.5-r*r/2)*e-a[0]*a

def d4_fisher_velocity(a):
    a=np.asarray(a,dtype=float);r2=float(a@a)
    if r2>=1:raise ValueError("parameter outside unit ball")
    # g^{-1}=(1-r^2)^2/(2*(d-1)) I when d=4
    return (1-r2)**2/6*d4_linear_gradient(a)

def d4_anisotropic_objective(a,eps=.15):
    a=np.asarray(a,dtype=float); r=np.linalg.norm(a)
    if r>=1: raise ValueError("parameter outside unit ball")
    if r==0:return 0.
    return d4_h1(r)*a[0]/r+eps*d4_h2(r)*(a[1]**2-a[2]**2)/r**2
