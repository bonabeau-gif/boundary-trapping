"""Uniform-reset mutation for cosine fitness; closure updates."""
import numpy as np
from .circle import fisher_velocity,moment_velocity

def reset_fisher(z,gamma):
    if gamma<0:raise ValueError("mutation rate must be nonnegative")
    return fisher_velocity(z,[1.]) - gamma*z*(1-abs(z)**2)

def reset_moment(z,gamma):
    if gamma<0:raise ValueError("mutation rate must be nonnegative")
    return moment_velocity(z,[1.]) - gamma*z

def fisher_equilibrium(gamma):
    """Unique real positive equilibrium for cosine selection with reset mutation."""
    if gamma<0:raise ValueError("mutation rate must be nonnegative")
    return np.sqrt(1+gamma*gamma)-gamma

def moment_equilibrium(gamma):
    return np.sqrt(1+gamma*gamma)-gamma
