import numpy as np

def harmonic_potential(x, omega=1.0):
    """
    Harmonisches Potential: V(x) = 0.5 * omega^2 * r^2
    x: numpy array mit shape (..., 3)
    """
    r_squared = np.sum(x**2, axis=-1)
    return 0.5 * omega**2 * r_squared
