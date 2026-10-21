"""Fisher-projected selection, moment closures, and validation models."""
from .circle import harmonic_extension, fisher_velocity, moment_velocity, exact_replicator, wrapped_cauchy, loss_cdf, loss_mean, transverse_second_derivative
from .cauchy import cauchy_density, lorentzian_expected_fitness, lorentzian_endpoint
__version__ = "0.1.0"
