import numpy as np
import matplotlib.pyplot as plt
from setup_inversion import set_true_m, unscale_param, get_inversion_indices, set_initial_conditions
from neptune_deterministic_methods import rk4_step

def build_neptune_vector(inverted_params_scaled):
    """
    Build full Neptune parameter vector from inverted parameters and m_0
    
    Args:
        inverted_params_scaled: The parameters being inverted (scaled)
    
    Returns:
        Full Neptune vector [mass, x, y, z, vx, vy, vz]
    """
    m_0 = set_true_m()
    neptune_full = m_0.copy()
    INVERT_INDICES = get_inversion_indices()
    inverted_params_scaled[0] = 10**inverted_params_scaled[0]  # Convert back to mass from log scale
    inverted_params_scaled = np.atleast_1d(inverted_params_scaled)
    inverted_unscaled = unscale_param(inverted_params_scaled)
    inverted_unscaled = np.atleast_1d(inverted_unscaled)
    
    for i, param_idx in enumerate(INVERT_INDICES):
        neptune_full[param_idx] = inverted_unscaled[i]
    
    return neptune_full

def predict_U(m, T: int = 190, dt: float = 1, z_scale_factor: int = 1) -> np.array:
    
    """
    Predict Uranus position based on initial conditions and parameter m, which contains Neptune's parameters at time T = 1775.

    Parameters
    ----------
    m : float or np.array
        The scaled parameters being inverted, which can be a single float or an array.
        
    Returns
    -------
    U_pred : np.array
        The predicted Uranus position in the format [x, y, z] scaled by z_scale_factor.
        The output is a 1D array of length 3*T, where T is the number of years.
    """

    initial_conditions = set_initial_conditions()
    bodies = {k: v.copy() for k, v in initial_conditions.items()}
    
    neptune_params = build_neptune_vector(m)
    bodies['Neptune'] = neptune_params.astype(np.float64)

    names = list(bodies.keys())
    n_bodies = len(names)
    masses = np.zeros(n_bodies, dtype=np.float64)
    positions = np.zeros((n_bodies, 3), dtype=np.float64)
    velocities = np.zeros((n_bodies, 3), dtype=np.float64)
    
    for i, name in enumerate(names):
        body_data = bodies[name]
        masses[i] = body_data[0]
        positions[i] = body_data[1:4]
        velocities[i] = body_data[4:7]
    
    total_momentum = np.sum(masses[1:, None] * velocities[1:], axis=0)
    velocities[0] = -total_momentum / masses[0]
    com = np.sum(masses[:, None] * positions, axis=0) / np.sum(masses)
    positions -= com
    
    uranus_positions = []
    uranus_idx = names.index('Uranus')
    
    t_days = 365 * T 
    num_steps = int(t_days / dt) - 1
    
    uranus_positions.append(positions[uranus_idx].copy())
    
    for step in range(num_steps):
        positions, velocities = rk4_step(positions, velocities, masses, dt)
        uranus_positions.append(positions[uranus_idx].copy())
    
    uranus_positions = np.array(uranus_positions)
    sampled = uranus_positions[::365]
    x_pred = sampled[:, 0]
    y_pred = sampled[:, 1]
    z_pred = sampled[:, 2] * z_scale_factor
    
    return np.concatenate([x_pred, y_pred, z_pred])

