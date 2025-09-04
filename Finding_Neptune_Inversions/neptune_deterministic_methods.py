'''
The following file consists of functions used in finding_neptune_via_deterministic_inv.ipynb 
'''

from numba import njit, jit
from tqdm import tqdm
import warnings
import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons        
from Finding_Neptune_Inversions.setup_inversion import get_inversion_indices, set_true_m, unscale_param, get_param_bounds, get_starting_points, get_param_scales, set_initial_conditions, get_arrow_data

warnings.filterwarnings('ignore')


@njit(fastmath=True, cache=True)
def acceleration(pos : np.ndarray, masses : np.ndarray) -> np.ndarray:
    """
    function to calculate the acceleration of each body due to gravitational forces from all other bodies.   

    Parameters
    ----------
    pos : np.array
        A numpy array of shape (n, 3) representing the positions of n bodies in 3D space.
    masses : np.array
        A numpy array of shape (n,) representing the masses of n bodies.

    Returns
    -------
    acc : np.array
        A numpy array of shape (n, 3) representing the accelerations of n bodies due to gravitational forces from all other bodies.
    """
    n = pos.shape[0]
    acc = np.zeros_like(pos)
    G = 0.0002959122082855911

    for i in range(n):
        for j in range(n):
            if i != j:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dz = pos[j, 2] - pos[i, 2]
                r_sq = dx*dx + dy*dy + dz*dz + 1e-12  # small epsilon to avoid divide-by-zero
                inv_r3 = 1.0 / (r_sq * np.sqrt(r_sq))
                factor = G * masses[j] * inv_r3
                acc[i, 0] += factor * dx
                acc[i, 1] += factor * dy
                acc[i, 2] += factor * dz
    return acc

@njit(fastmath=True, cache=True)
def rk4_step(pos : np.ndarray, vel : np.ndarray, masses : np.ndarray, dt : float) -> tuple:
    """
    Perform a single Runge-Kutta 4th order step to update positions and velocities of celestial bodies. The method uses the current positions and velocities, calculates the accelerations, and updates the positions and velocities accordingly.

    Parameters
    ----------
    pos : np.ndarray
        positions of the bodies in 3D space, shape (n, 3).
        
    vel : np.ndarray
        velocities of the bodies in 3D space, shape (n, 3).
    masses : np.ndarray
        masses of the bodies, shape (n,).
    dt : float
        time step for the simulation.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing:
        - new_pos :
            updated positions of the bodies after the time step, shape (n, 3).
        - new_vel :
            updated velocities of the bodies after the time step, shape (n, 3). 
    """
    a1 = acceleration(pos, masses)
    v1 = vel
    p2 = pos + 0.5 * dt * v1
    v2 = vel + 0.5 * dt * a1
    a2 = acceleration(p2, masses)
    p3 = pos + 0.5 * dt * v2
    v3 = vel + 0.5 * dt * a2
    a3 = acceleration(p3, masses)
    p4 = pos + dt * v3
    v4 = vel + dt * a3
    a4 = acceleration(p4, masses)
    
    new_pos = pos + dt * (v1 + 2.0*v2 + 2.0*v3 + v4) / 6.0
    new_vel = vel + dt * (a1 + 2.0*a2 + 2.0*a3 + a4) / 6.0
    
    return new_pos, new_vel

def run_simulation(initial_conditions: dict = None, T: float = 10, dt: float = 1, plot : bool = True, plot_only : list = None):
    """
    Run a simulation of the solar system 
    
    Parameters
    ----------
    initial_conditions : dict
        a dictionary containing the initial conditions of the celestial bodies, where each key is the name of a body and the value is a list containing its mass and initial position and velocity components [mass, x, y, z, vx, vy, vz]. 
    T : float, optional
    dt : float, optional
    plot : bool, optional
    plot_only : list, optional

    Returns
    -------
    trajectories : dict
        A dictionary containing the trajectories of each celestial body, where each key is the name of a body and the value is a numpy array of shape (num_steps, 3) representing the x, y, z positions of the body at each time step.
        e.g. {
              'Mercury': np.array([[x1, y1, z1], [x2, y2, z2], ...]),...
              }
    """
    G = 0.0002959122082855911  # Gravitational constant in AU^3 / (day^2 * solar mass)
    
    if initial_conditions is None:
        bodies = {
            'Sun':     [1.0,7.905568646423392E-04, -4.264417232280915E-03, -5.801050077663720E-05,  5.661919461064193E-06, -3.329221898768886E-06, -1.332363131747294E-07],
            'Mercury': [1.652e-7, -3.807707219093185E-01, -1.921733743242082E-01,  1.989005978017707E-02,  6.640479515650169E-03, -2.398063254332634E-02, -2.564565887796372E-03],
            'Venus':   [2.447e-6, 1.280683608568111E-01, -7.204436733949646E-01, -1.676915146013050E-02,  1.978145114380488E-02,  3.462734858396583E-03, -1.099876178640022E-03],
            'Earth':   [3.003e-6, -2.312108463257164E-01,  9.512434439348221E-01,  4.175708629657505E-04, -1.700259811686698E-02, -4.123625854835961E-03, -2.671125625686586E-06],
            'Mars':    [3.227e-7, -1.161863685676914E+00,  1.172396515331657E+00,  5.363743556489492E-02, -9.398564528128208E-03, -8.659237404387224E-03,  5.494330200022026E-05],
            'Jupiter': [9.545e-4, 3.243139912281485E+00,  3.792387174781622E+00, -8.808502227918256E-02, -5.821367401944318E-03,  5.261546432625223E-03,  1.099868754978308E-04],
            'Saturn':  [2.857e-4, -9.478453610515910E+00, -9.743812139018955E-01,  3.918886540177342E-01,  2.645573154424449E-04, -5.565302115104686E-03,  8.852958646059887E-05],
            'Uranus':  [4.366e-5, 8.510442128580516E+00,  1.743332182201955E+01, -4.523020170261750E-02, -3.566647203548249E-03,  1.544035350391986E-03,  5.235164486852619E-05],
            'Neptune': [5.151e-5, -3.005586533802299E+01,  3.108513809535168E+00,  6.280746172524112E-01, 
                        -3.404729678559355E-04, -3.103001549555015E-03,  7.188313350941659E-05],
        }
    else:
        bodies = initial_conditions

    names = list(bodies.keys())
    n = len(names)
    masses = np.array([bodies[name][0] for name in names])
    positions = np.array([bodies[name][1:4] for name in names], dtype=float)
    velocities = np.array([bodies[name][4:7] for name in names], dtype=float)

    com = np.sum(masses[:, None] * positions, axis=0) / np.sum(masses)
    positions -= com
    
    # Center of mass velocity correction, since we want to simulate in an inertial frame
    # Total momentum conservation - should be zero for the whole system as we want inertial frame for reference
    total_momentum = np.sum(masses[:, None] * velocities, axis=0)
    # Distribute momentum correction proportionally to masses
    velocities -= total_momentum / np.sum(masses)

    num_steps = int(365 * T / dt)  # T=190 years, dt=1 day -> 69,350 steps
    trajectories = {name: [] for name in names}
    for i, name in enumerate(names):
        trajectories[name].append(positions[i].copy())
    for _ in tqdm(range(num_steps), desc='Computing trajectories...'):
        positions, velocities = rk4_step(positions, velocities, masses, dt)
        for i, name in enumerate(names):
            trajectories[name].append(positions[i].copy())
    
    for name in names:
        trajectories[name] = np.array(trajectories[name])
        
    if plot:    
        plt.figure(figsize=(7, 7))
        for name in plot_only or names:
            if name != 'Sun':
                plt.plot(trajectories[name][:, 0], trajectories[name][:, 1], label=name)
        plt.plot(0, 0, 'yo', label='Sun')
        plt.xlabel('x (AU)')
        plt.ylabel('y (AU)')
        plt.title('Planetary Orbits Around the Sun')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()
        
    return trajectories

def generate_synthetic_data(T : int, dt :  float, z_scaling : bool, add_noise : bool, noise_level : np.array = None) -> np.array:
    
    """
    Generate Synthetica data for Uranus which can mimic the actual data from mASA.
    
    Parameters
    ----------
    T : int
        The number of years to generate synthetic data for.
    dt : float
        The time step in years for the simulation.
    z_scaling : bool
        Whether to scale the Z coordinate to match the magnitude of X and Y.
    add_noise : bool
        Whether to add noise to the synthetic data.
    noise_level : np.array, optional
        The standard deviation of the noise to be added to X, Y, and Z coordinates, should be a 3-element array.
    Returns
    -------
    U_true : np.array
        The synthetic data for Uranus, concatenated X, Y, and Z coordinates, shaped as a 1D array of length 3*T.    
    """
    
    initial_conditions = {
            'Sun':     [1.0, 7.905568646423392E-04, -4.264417232280915E-03, -5.801050077663720E-05,  5.661919461064193E-06, -3.329221898768886E-06, -1.332363131747294E-07],
            'Mercury': [1.652e-7, -3.807707219093185E-01, -1.921733743242082E-01,  1.989005978017707E-02,  6.640479515650169E-03, -2.398063254332634E-02, -2.564565887796372E-03],
            'Venus':   [2.447e-6, 1.280683608568111E-01, -7.204436733949646E-01, -1.676915146013050E-02,  1.978145114380488E-02,  3.462734858396583E-03, -1.099876178640022E-03],
            'Earth':   [3.003e-6, -2.312108463257164E-01,  9.512434439348221E-01,  4.175708629657505E-04, -1.700259811686698E-02, -4.123625854835961E-03, -2.671125625686586E-06],
            'Mars':    [3.227e-7, -1.161863685676914E+00,  1.172396515331657E+00,  5.363743556489492E-02, -9.398564528128208E-03, -8.659237404387224E-03,  5.494330200022026E-05],
            'Jupiter': [9.545e-4, 3.243139912281485E+00,  3.792387174781622E+00, -8.808502227918256E-02, -5.821367401944318E-03,  5.261546432625223E-03,  1.099868754978308E-04],
            'Saturn':  [2.857e-4, -9.478453610515910E+00, -9.743812139018955E-01,  3.918886540177342E-01,  2.645573154424449E-04, -5.565302115104686E-03,  8.852958646059887E-05],
            'Uranus':  [4.366e-5, 8.510442128580516E+00,  1.743332182201955E+01, -4.523020170261750E-02, -3.566647203548249E-03,  1.544035350391986E-03,  5.235164486852619E-05],
                'meptune': [5.151e-5, -3.005586533802299E+01,  3.108513809535168E+00,  6.280746172524112E-01, 
                    
                        -3.404729678559355E-04, -3.103001549555015E-03,  7.188313350941659E-05]
        }
    synthetic_trajectories = run_simulation(initial_conditions=initial_conditions, T=T-1, dt=dt, plot=False)

    synthetic_data = synthetic_trajectories['Uranus'][::int(365/dt)]

    x_data = synthetic_data[:, 0]
    y_data = synthetic_data[:, 1] 
    z_data = synthetic_data[:, 2]

    print("\nSynthetic data ranges:")
    print(f"X: {x_data.min():.3f} to {x_data.max():.3f} (range: {x_data.max() - x_data.min():.3f})")
    print(f"Y: {y_data.min():.3f} to {y_data.max():.3f} (range: {y_data.max() - y_data.min():.3f})")
    print(f"Z: {z_data.min():.6f} to {z_data.max():.6f} (range: {z_data.max() - z_data.min():.6f})")
    
    if add_noise:
        x_noise = np.random.normal(0, noise_level[0], size=x_data.shape)
        y_noise = np.random.normal(0, noise_level[1], size=y_data.shape)
        z_noise = np.random.normal(0, noise_level[2], size=z_data.shape)
        x_data += x_noise
        y_data += y_noise
        z_data += z_noise
        
        print("\nNoisy synthetic data ranges:")
        print(f"X: {x_data.min():.3f} to {x_data.max():.3f} (range: {x_data.max() - x_data.min():.3f})")
        print(f"Y: {y_data.min():.3f} to {y_data.max():.3f} (range: {y_data.max() - y_data.min():.3f})")
        print(f"Z: {z_data.min():.6f} to {z_data.max():.6f} (range: {z_data.max() - z_data.min():.6f})")
    
    if z_scaling:
        z_data_scaled = z_data * 100
        print(f"\nAfter scaling Z by 100:")
        print(f"Z_scaled: {z_data_scaled.min():.3f} to {z_data_scaled.max():.3f} (range: {z_data_scaled.max() - z_data_scaled.min():.3f})")
        U_synthetic = np.concatenate([x_data, y_data, z_data_scaled])
    else:
        U_synthetic = np.concatenate([x_data, y_data, z_data])  
    
    return U_synthetic

def build_neptune_vector(inverted_params_scaled):
    
    """
    Build full Neptune parameter vector from inverted parameters and m_0
    
    Args:
        inverted_params_scaled: The parameters being inverted (scaled)
    
    Returns:
        Full Neptune vector [mass, x, y, z, vx, vy, vz], the missing parameters are taken from m_0.
    """
    m_0 = set_true_m()
    neptune_full = m_0.copy()
    INVERT_INDICES = get_inversion_indices()
    inverted_params_scaled = np.atleast_1d(inverted_params_scaled)
    inverted_unscaled = unscale_param(inverted_params_scaled)
    inverted_unscaled = np.atleast_1d(inverted_unscaled)
    
    for i, param_idx in enumerate(INVERT_INDICES):
        neptune_full[param_idx] = inverted_unscaled[i]
    
    return neptune_full

def predict_U(m, T : int = 190, dt : float = 1, z_scale_factor : int = 1) -> np.array:
    
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

def residual(m, U_true, alpha : float = 0, T : int = 190, dt : float = 1):
    
    """
    Calculate the residual between predicted and true Uranus positions.

    Returns
    -------
    residual : np.array
        The difference between predicted Uranus positions and true positions.
        This is a 1D array of length 3*T, where T is the number of years, first T elements are X, next T are Y, and last T are Z.
    """
    
    U_pred = predict_U(m, T=T, dt=dt)
    
    if alpha > 0:
        reg_term = alpha * m
        return np.concatenate([(U_pred - U_true), reg_term])
    
    return (U_pred - U_true)

def jacobian(m, U_true, alpha : float = 0, T : int = 190, dt : float = 1):
    
    """
    Calculate the Jacobian matrix of the residual function with respect to the parameters m.

    Returns
    -------
    J : np.array
        The Jacobian matrix, where each column corresponds to the partial derivative of the residual with respect to each parameter in m.
        The shape is (3*T, len(m)), where T is the number of years.
        The first T elements correspond to the X component, the next T to Y, and the last T to Z.
    """
    
    epsilon = 1e-12
    m_array = np.atleast_1d(m)
    J = np.zeros((len(U_true), len(m_array)), dtype=np.float64)
    U_base = predict_U(m, T=T, dt=dt)
    
    for i in range(len(m_array)):
        m_plus = m_array.copy()
        m_plus[i] += epsilon
        U_plus = predict_U(m_plus, T=T, dt=dt)
        J[:, i] = (U_plus - U_base) / epsilon
        
    if alpha > 0:
        return np.vstack([J, np.eye(len(m)) * alpha])  # Add regularization term to Jacobian
    # print(np.diag(J.T @ J))       # Uncomment to see diagonal of J.T@J
    # print(f"Jacobian condition number: {np.linalg.cond(J.T@J):.2e}")  # Uncomment to see condition number of J.T@J
    return J

def set_lcurve_inversion_params() -> tuple:
    '''Set parameters for the L-curve inversion.'''
    U_true = generate_synthetic_data(T = 190, dt = 1, z_scaling = False, 
                                 add_noise = True, 
                                 noise_level = np.array([0.001, 0.001, 0.00001]))
    T = 190
    dt = 1
    alphas = np.logspace(-4, 2, 10)
    return U_true, T, dt, alphas

def callback_func(inv_result, i) -> tuple:
    """
    Callback function to process inversion results and print norms.
    
    Parameters
    ----------
    inv_result : InversionResult
        The result of the inversion containing the model and other information.
    i : int
        The index of the alpha value used for this inversion.

    Returns
    -------
    tuple
        A tuple containing the residual norm and solution norm for the inversion.
    -----------
    """
    U_true, T, dt, alphas = set_lcurve_inversion_params()
    result = inv_result.model
    final_pred = predict_U(result, T=T, dt=dt)
    residual_norm = np.linalg.norm(U_true - final_pred)
    solution_norm = np.linalg.norm(result)
    print(f"Alpha {alphas[i]:.2e} - Residual norm: {residual_norm:.6f}, Solution norm: {solution_norm:.6f}")
    return residual_norm, solution_norm

def plot_uranus_orbits(predicted_uranus_trajectory, U_true, T, z_scale_factor : float = 1):


    x_pred = predicted_uranus_trajectory[:T]
    y_pred = predicted_uranus_trajectory[T:2*T]
    z_pred = predicted_uranus_trajectory[2*T:] / z_scale_factor  # Scale Z back to original units

    x_true = U_true[:T]
    y_true = U_true[T:2*T]
    z_true = U_true[2*T:] / z_scale_factor  # Scale Z back to original units

    fig, axs = plt.subplots(1, 3, figsize=(18, 8))

    # XY plane
    axs[0].plot(x_pred, y_pred, label='Predicted', color='blue')
    axs[0].plot(x_true, y_true, 'o', label='True', color='red')
    axs[0].set_xlabel("X (AU)")
    axs[0].set_ylabel("Y (AU)")
    axs[0].set_title("XY Plane")
    axs[0].legend()
    axs[0].grid(True)

    # YZ plane
    axs[1].plot(y_pred, z_pred, label='Predicted', color='blue')
    axs[1].plot(y_true, z_true, 'o', label='True', color='red')
    axs[1].set_xlabel("Y (AU)")
    axs[1].set_ylabel("Z (AU)")
    axs[1].set_title("YZ Plane")
    axs[1].legend()
    axs[1].grid(True)

    # ZX plane
    axs[2].plot(z_pred, x_pred, label='Predicted', color='blue')
    axs[2].plot(z_true, x_true, 'o', label='True', color='red')
    axs[2].set_xlabel("Z (AU)")
    axs[2].set_ylabel("X (AU)")
    axs[2].set_title("ZX Plane")
    axs[2].legend()
    axs[2].grid(True)

    plt.suptitle("Uranus Trajectory Projections (Observed vs Predicted)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_neptune_orbits(result_unscaled, initial_conditions, T, dt):

    m_0 = set_true_m()
    initial_conditions['Neptune'] = result_unscaled
    synthetic_trajectories = run_simulation(initial_conditions=initial_conditions, T=T, dt=dt, plot=False)
    synthetic_neptune_data = synthetic_trajectories['Neptune']

    initial_conditions['Neptune'] = np.array([(5.151e-5), *m_0[1:7]], dtype=np.float64)
    synthetic_trajectories_true = run_simulation(initial_conditions=initial_conditions, T=T, dt=dt, plot=False)
    true_neptune_data = synthetic_trajectories_true['Neptune']

    num_arrows = 8

    x_est, y_est, dx_est, dy_est = get_arrow_data(synthetic_neptune_data, num_arrows, arrow_length=0.8)
    x_true, y_true, dx_true, dy_true = get_arrow_data(true_neptune_data, num_arrows, arrow_length=0.8)

    plt.figure(figsize=(12, 8))

    plt.plot(synthetic_neptune_data[:, 0], synthetic_neptune_data[:, 1], 
            label='Estimated Neptune Orbit', color='crimson', linewidth=2.5, alpha=0.8)
    plt.plot(true_neptune_data[:, 0], true_neptune_data[:, 1], 
            label='True Neptune Orbit', color='navy', linestyle='--', linewidth=2.5, alpha=0.8)

    plt.quiver(
        x_est, y_est, dx_est, dy_est,
        angles='xy', scale_units='xy', scale=1, 
        width=0.004, 
        headwidth=4,  
        headlength=5,
        headaxislength=4, 
        color='crimson', alpha=0.9,
        label='Estimated Direction'
    )

    plt.quiver(
        x_true, y_true, dx_true, dy_true,
        angles='xy', scale_units='xy', scale=1,
        width=0.004,  # Thinner shaft
        headwidth=4,  # Bigger arrow head
        headlength=5, # Longer arrow head
        headaxislength=4,  # Arrow head axis length
        color='darkblue', alpha=0.9,
        label='True Direction'
    )

    plt.scatter(synthetic_neptune_data[0, 0], synthetic_neptune_data[0, 1], 
            s=100, c='crimson', marker='o', edgecolor='white', linewidth=2, 
            label='Estimated Start', zorder=5)
    plt.scatter(synthetic_neptune_data[-1, 0], synthetic_neptune_data[-1, 1], 
            s=100, c='crimson', marker='s', edgecolor='white', linewidth=2, 
            label='Estimated End', zorder=5)

    plt.scatter(true_neptune_data[0, 0], true_neptune_data[0, 1], 
            s=100, c='navy', marker='o', edgecolor='white', linewidth=2, 
            label='True Start', zorder=5)
    plt.scatter(true_neptune_data[-1, 0], true_neptune_data[-1, 1], 
            s=100, c='navy', marker='s', edgecolor='white', linewidth=2, 
            label='True End', zorder=5)

    plt.annotate('START', xy=(synthetic_neptune_data[0, 0], synthetic_neptune_data[0, 1]), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold', color='crimson',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.annotate('END', xy=(synthetic_neptune_data[-1, 0], synthetic_neptune_data[-1, 1]), 
                xytext=(-10, -10), textcoords='offset points',
                fontsize=10, fontweight='bold', color='crimson',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('X [AU]', fontsize=12, fontweight='bold')
    plt.ylabel('Y [AU]', fontsize=12, fontweight='bold')
    plt.axis('equal')
    plt.title('Neptune Orbit Comparision \n(inversion parameters vs True Parameters)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Estimated orbit: {len(synthetic_neptune_data)} points")
    print(f"True orbit: {len(true_neptune_data)} points")
    print(f"Number of direction arrows per orbit: {num_arrows}")
    print(f"Orbital direction: Start (circle) → End (square)")
    
def get_actual_data(z_scaling : bool = False, T : int = 190):
    
    """
    Get actual data from NASA for the planets in the solar system.
    
    Parameters
    ----------
    z_scaling : bool, optional
        If True, scales the z-coordinate by a factor of 1000. Default is False.
    
    T : int, optional
        The number of years for which to get the data. Default is 190.
    """
    
    T_end = 1775 + T -1
    T_end = str(T_end) + '-01-01'
    
    obj = Horizons(id='799', location='500@0', epochs={'start':'1775-01-01', 'stop':T_end, 'step':'1y'}, id_type='majorbody')

    vec = obj.vectors()

    data_lines = []
    for row in vec:
        jd = float(row['datetime_jd'])
        date = row['datetime_str']
        x = float(row['x'])
        y = float(row['y'])
        z = float(row['z'])
        
        line = f"{jd:.9f}, {date}, {x: .15E}, {y: .15E}, {z: .15E},"
        data_lines.append(line)

    data = "\n".join(data_lines)
    
    observed_positions = []
    for line in data.strip().split('\n'):
        parts = line.split(',')
        x = float(parts[2])
        y = float(parts[3])
        z = float(parts[4])
        
        observed_positions.append([x, y, z])

    observed_positions = np.array(observed_positions)

    observed_positions = observed_positions[:T]

    print("Original data ranges:")
    x_data_actual = observed_positions[:, 0]
    y_data_actual = observed_positions[:, 1] 
    z_data_actual = observed_positions[:, 2]

    print(f"X: {x_data_actual.min():.3f} to {x_data_actual.max():.3f} (range: {x_data_actual.max() - x_data_actual.min():.3f})")
    print(f"Y: {y_data_actual.min():.3f} to {y_data_actual.max():.3f} (range: {y_data_actual.max() - y_data_actual.min():.3f})")
    print(f"Z: {z_data_actual.min():.6f} to {z_data_actual.max():.6f} (range: {z_data_actual.max() - z_data_actual.min():.6f})")

    if z_scaling:
        # Scale Z to make it similar magnitude to X, Y
        z_scale_factor = 100
        z_data_actual_scaled = z_data_actual * z_scale_factor
        print(f"\nAfter scaling Z by {z_scale_factor}:")
        print(f"Z_scaled: {z_data_actual_scaled.min():.3f} to {z_data_actual_scaled.max():.3f} (range: {z_data_actual_scaled.max() - z_data_actual_scaled.min():.3f})")
    else:
        # Do not scale Z, keep it as is
        z_data_actual_scaled = z_data_actual
        print("\nZ scaling is disabled, using original Z values.")
        
    U_true = np.concatenate([x_data_actual, y_data_actual, z_data_actual_scaled])
    
    return U_true
