import numpy as np

def apply_branching(positions, potential_fn, dt, E_T, M_tilde, alpha=0.01):
    """
    Führt das Branching gemäß dem DMC-Algorithmus durch.
    Verwendet dynamisch angepasstes E_T basierend auf Populationskontrolle.
    """
    V = potential_fn(positions)
    weights = np.exp(-(V - E_T) * dt)

    new_positions = []
    for i in range(len(positions)):
        m = int(weights[i] + np.random.rand())
        new_positions.extend([positions[i]] * m)

    new_positions = np.array(new_positions)
    M = len(new_positions)

    # Dynamische Anpassung von E_T
    if M > 0:
        running_M = 0.9 * M_tilde + 0.1 * M
        E_T_new = E_T + alpha * np.log(M_tilde / running_M)
    else:
        E_T_new = E_T  # Fallback: kein Update, Population leer

    return new_positions, E_T_new