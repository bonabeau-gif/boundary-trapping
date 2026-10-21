"""Analytical circular replicator and wrapped-Cauchy closures.

Fourier convention: f(theta)=c0+sum_{k>=1} Re(A[k-1]*exp(i*k*theta)).
For z=u+iv, circular moments E[exp(i*k*theta)]=z**k.
"""
import numpy as np
from scipy.integrate import solve_ivp


def wrapped_cauchy(theta, z):
    z=complex(z)
    if abs(z)>=1: raise ValueError("parameter must be strictly inside disk")
    return (1-abs(z)**2)/(2*np.pi*np.abs(np.exp(1j*np.asarray(theta))-z)**2)


def harmonic_extension(z, harmonics, constant=0.):
    return float(constant+sum(np.real(c*z**k) for k,c in enumerate(harmonics,1)))


def complex_gradient(z, harmonics):
    """J_u+i J_v, with J(z)=Re(sum A_k z^k)."""
    return np.conj(sum(k*c*z**(k-1) for k,c in enumerate(harmonics,1)))


def harmonic_conjugate(z, harmonics):
    return float(np.imag(sum(c*z**k for k,c in enumerate(harmonics,1))))


def fisher_velocity(z, harmonics):
    """Physical-time Fisher-projected replicator velocity."""
    return 0.5*(1-abs(z)**2)**2*complex_gradient(z,harmonics)


def moment_velocity(z, harmonics):
    """First-moment closure. Constant fitness cancels identically."""
    return 0.5*(1-abs(z)**2)*sum(np.conj(c)*np.conj(z)**(k-1) for k,c in enumerate(harmonics,1))


def cosine_fisher(t,y):
    z=complex(y[0],y[1]); dz=fisher_velocity(z,[1.])
    return np.array([dz.real,dz.imag])


def integrate_fisher(z0,harmonics,t_end=100.,**kw):
    def rhs(_,y):
        z=complex(*y); dz=fisher_velocity(z,harmonics)
        return [dz.real,dz.imag]
    return solve_ivp(rhs,(0,t_end),[z0.real,z0.imag],rtol=kw.pop('rtol',1e-9),atol=kw.pop('atol',1e-11),**kw)


def exact_replicator(theta,t,initial_density,fitness):
    """Continuous density, numerically normalized on a periodic uniform grid."""
    theta=np.asarray(theta); p=np.asarray(initial_density,dtype=float)
    logw=np.log(p)+t*np.asarray(fitness,dtype=float)
    w=np.exp(logw-np.max(logw))
    return w / (w.sum()*(2*np.pi/len(theta)))


def projected_cosine_endpoint(z0):
    return np.sqrt(1-z0.imag**2)+1j*z0.imag


def transverse_second_derivative(z):
    """Exact d^2/dt^2 E[sin theta] at t=0 under cos theta selection."""
    return -z.imag*(1-abs(z)**2)/4


def loss_mean():
    return 1-8/(3*np.pi)


def loss_cdf(ell):
    """P(1-sqrt(1-v^2) <= ell), uniform area in unit disk."""
    val=np.asarray(ell,dtype=float)
    x=np.clip(val,0,1); a=np.sqrt(np.maximum(0,2*x-x*x))
    out=2/np.pi*(np.arcsin(a)+a*np.sqrt(np.maximum(0,1-a*a)))
    return float(out) if out.ndim==0 else out
