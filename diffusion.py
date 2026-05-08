import numpy as np

def diffuse_walkers(positions, step_size):
    """
    Führt einen Diffusionsschritt für alle Walker im 3D-Raum durch.

    Parameter:
    - positions: numpy-Array der Form (N, 3), aktuelle Positionen der Walker
    - step_size: float, entspricht Δt (Zeitintervall)

    Rückgabe:
    - neue Positionen nach einem Schritt
    """
    num_walkers = positions.shape[0]
    
    # Gaußverteilte Zufallsschritte ~ N(0, sqrt(2 * step_size))
    steps = np.random.normal(loc=0.0, scale=np.sqrt(2 * step_size), size=(num_walkers, 3))
    
    # Neue Positionen
    new_positions = positions + steps
    return new_positions
