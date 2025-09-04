import numpy as np

def get_starting_points():
    STARTING_POINTS = [((1.02e26)/(1.9885e30)), 
                   -2.991585533802299E+01, 
                   3.1455510809535168E+00, 
                   1.050744172524112E-01, 
                   -2.804329678559355E-04, 
                   -3.11021549555015E-03, 
                   4.187313350941659E-05
                   ]
    return STARTING_POINTS

def set_true_m():
    m_0 = np.array([
    5.151e-5,                    # mass (solar masses)
    -3.005586533802299E+01,      # x position (AU)
    3.108513809535168E+00,       # y position (AU) 
    6.280746172524112E-01,       # z position (AU)
    -3.404729678559355E-04,      # x velocity (AU/day)
    -3.103001549555015E-03,      # y velocity (AU/day)
    7.188313350941659E-05        # z velocity (AU/day)
    ])
    return m_0

def get_inversion_indices():
    """Get the indices of the parameters to be inverted"""
    # CONFIGURATION: Which parameters to invert for (indices: 0=mass, 1=x, 2=y, 3=z, 4=vx, 5=vy, 6=vz)
    # INVERT_INDICES = [0, 1, 2, 3]  # Mass + position  
    # INVERT_INDICES = [0, 4, 5, 6]  # Mass + velocities
    # INVERT_INDICES = [1, 2, 3]     # Only position
    INVERT_INDICES = list(range(7)) # All parameters

    return INVERT_INDICES

def get_param_scales():
    """Get the parameter scales for inversion"""
    
    # Parameter bounds [lower, upper] for each parameter being inverted, to be used ONLY when using trf method in inversion_options
    PARAM_BOUNDS = [
        [1e-8, 1e-3],  
        [-50, 50],  
        [-50, 50], 
        [-10, 10],    
        [-0.001, 0.001],  
        [-0.01, 0.01],  
        [-0.001, 0.001]   
    ]
    # CONFIGURATION: Which parameters to invert for (indices: 0=mass, 1=x, 2=y, 3=z, 4=vx, 5=vy, 6=vz)
    # INVERT_INDICES = [0, 1, 2, 3]  # Mass + position  
    # INVERT_INDICES = [0, 4, 5, 6]  # Mass + velocities
    # INVERT_INDICES = [1, 2, 3]     # Only position
    INVERT_INDICES = get_inversion_indices()

    PARAM_SCALES = []

    # Scaling factors for each parameter to normalize the inversion
    # These are based on the diagonal of J.T@J, which gives an idea of the relative scales of the parameters.

    S = (np.sqrt([1.28428575e+01, 2.35833841e-02,1.89992459e-02 ,4.22416638e-01
    ,6.39191788e+05 ,1.44721770e+06, 4.53527575e+03]))

    for i in INVERT_INDICES:
        PARAM_SCALES.append(S[i])
        
    return PARAM_SCALES

def get_param_bounds():
    PARAM_BOUNDS = [
    [1e-8, 1e-3],  
    [-50, 50],  
    [-50, 50], 
    [-10, 10],    
    [-0.001, 0.001],  
    [-0.01, 0.01],  
    [-0.001, 0.001]   
    ]
    return PARAM_BOUNDS

def validate_config():
    """Validate that configuration arrays have consistent lengths"""
    
    INVERT_INDICES = get_inversion_indices()
    PARAM_SCALES = get_param_scales()
    PARAM_BOUNDS = get_param_bounds()
    STARTING_POINTS = get_starting_points()
    
    n_params = len(INVERT_INDICES)
    assert len(STARTING_POINTS) == n_params, f"STARTING_POINTS length {len(STARTING_POINTS)} != INVERT_INDICES length {n_params}"
    assert len(PARAM_BOUNDS) == n_params, f"PARAM_BOUNDS length {len(PARAM_BOUNDS)} != INVERT_INDICES length {n_params}"
    assert len(PARAM_SCALES) == n_params, f"PARAM_SCALES length {len(PARAM_SCALES)} != INVERT_INDICES length {n_params}"
    
    param_names = ['mass', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    print(f"Inverting for: {[param_names[i] for i in INVERT_INDICES]}")
    print(f"Starting points: {STARTING_POINTS}")
    
def scale_param(m):
    """Scale parameters - handle both scalar and array inputs"""
    
    PARAM_SCALES = get_param_scales()
    if np.isscalar(m):
        return PARAM_SCALES[0] * m if len(PARAM_SCALES) == 1 else set_true_m()
    else:
        m_array = np.asarray(m)
        if m_array.size == 1:
            return PARAM_SCALES[0] * m_array.item()
        else:
            return np.array([PARAM_SCALES[i] * m_array[i] for i in range(len(m_array))])

def unscale_param(m_scaled):
    """
    Unscales the parameters from their scaled values.

    Returns
    -------
    m_unscaled : float or np.array
        The unscaled parameters, either as a single float or an array.
    """
    PARAM_SCALES = get_param_scales()
    if np.isscalar(m_scaled):
        return m_scaled / PARAM_SCALES[0] if len(PARAM_SCALES) == 1 else m_scaled
    else:
        m_array = np.asarray(m_scaled)
        if m_array.size == 1:
            return m_array.item() / PARAM_SCALES[0]
        else:
            return np.array([m_array[i] / PARAM_SCALES[i] for i in range(len(m_array))])

def set_initial_conditions():
    global initial_conditions
    initial_conditions =  {
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
    return initial_conditions

def get_arrow_data(data, num_arrows, arrow_length=0.5):
    
    """
    Extracts arrow data from the given orbit data for plotting direction arrows.

    Returns
    -------
    x, y, dx_norm, dy_norm : np.array
        Arrays containing the x and y coordinates of the arrow starting points, and the normalized dx and dy components for the arrows.
        
    Parameters
    ----------
    data : np.array
        The orbit data as a 2D array with shape (N, 2) where N is the number of points and each point has x and y coordinates.
    num_arrows : int
        The number of arrows to extract from the orbit data.
    arrow_length : float, optional
        The length of the arrows to be plotted. Default is 0.5.

    """

    idx = np.linspace(0, len(data) - 2, num_arrows, dtype=int)
    x = data[idx, 0]
    y = data[idx, 1]
    dx = data[idx + 1, 0] - data[idx, 0]
    dy = data[idx + 1, 1] - data[idx, 1]
    
    norms = np.sqrt(dx**2 + dy**2)
    norms = np.where(norms == 0, 1, norms)
    dx_norm = (dx / norms) * arrow_length
    dy_norm = (dy / norms) * arrow_length
    
    return x, y, dx_norm, dy_norm
