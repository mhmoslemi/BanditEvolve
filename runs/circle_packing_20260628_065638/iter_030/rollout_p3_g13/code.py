import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize a grid-like layout with stochastic geometric perturbations and advanced clustering
    xs = []
    ys = []
    radii = []
    
    # First, construct a grid with some spatial diversity
    for i in range(n):
        col = i % cols
        row_idx = i // cols
        base_x = col + 0.5
        base_y = row_idx + 0.5
        
        # Grid-based center: (col + 0.5) / cols, (row_idx + 0.5) / rows, but now with row-wise scaling
        x_base = (col + 0.3) / cols * 1.2  # Slight row-wise expansion for better spacing
        y_base = (row_idx + 0.2) / rows * 1.1  # Row-wise scaling for non-uniform height
            
        x = x_base + np.random.uniform(-0.1, 0.1) * (0.5 / cols) * (0.9 + np.random.rand())  # Adaptive offset based on grid spacing
        y = y_base + np.random.uniform(-0.1, 0.1) * (0.5 / rows) * np.random.rand()  # Row-specific spatial randomness
            
        # Alternate row staggering to prevent clustering
        if row_idx % 3 <= 1:  # More frequent staggering for better density control
            x += 0.5 / cols * (0.7 + np.random.rand() * 0.3)  # Stagger with adaptive random scaling
            
        # Apply geometric distortion to break symmetry and induce dynamic layout
        x = x + np.random.normal(0, 0.005)  # Gaussian jitter
        y = y + np.random.normal(0, 0.005)
        
        xs.append(np.clip(x, 0.001, 0.999))  # Clipping to maintain within bounds
        ys.append(np.clip(y, 0.001, 0.999))
        # Base radii: based on grid spacing and spatial distortion coefficient
        radius = (0.4 / cols) * (1.0 - (np.random.rand() * 0.3)) - 1e-3  # Adjusted and reduced initial radius to improve packing
        radii.append(radius)
    
    # Prepare the initial vector
    v0 = np.zeros(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.array(radii)
    
    # Create bounds list of length 3*n
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Radius lower bound set for stricter feasibility

    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Vectorized constraint construction
    # Use a functional closure with capturing i instead of lambda-based closures to fix scoping issues
    cons = []
    for i in range(n):
        # Left side: x - radius >= 0
        def constraint_left(v, i=i):
            return v[3*i] - v[3*i + 2]
        cons.append({"type": "ineq", "fun": constraint_left})
        # Right side: 1.0 - (x + radius) >= 0
        def constraint_right(v, i=i):
            return 1.0 - v[3*i] - v[3*i + 2]
        cons.append({"type": "ineq", "fun": constraint_right})
        # Bottom: y - radius >= 0
        def constraint_bottom(v, i=i):
            return v[3*i + 1] - v[3*i + 2]
        cons.append({"type": "ineq", "fun": constraint_bottom})
        # Top: 1.0 - (y + radius) >= 0
        def constraint_top(v, i=i):
            return 1.0 - v[3*i + 1] - v[3*i + 2]
        cons.append({"type": "ineq", "fun": constraint_top})

    # Overlap constraints with improved vectorization and gradient handling
    for i in range(n):
        for j in range(i + 1, n):
            # Vectorizable constraint function: distance squared - sum of radii squared
            # Avoid lambda capturing via explicit parameter capture with helper function
            def constraint_overlap(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i + 1] - v[3*j + 1]
                radius_i = v[3*i + 2]
                radius_j = v[3*j + 2]
                return dx**2 + dy**2 - (radius_i + radius_j)**2  # Negative to invert constraint direction for optimization
            cons.append({"type": "ineq", "fun": constraint_overlap})

    # First optimization to converge on a basic solution
    first_opt_options = {
        "maxiter": 1800,
        "ftol": 1e-12,
        "gtol": 1e-12,  # Stricter gradient tolerance
        "eps": 1e-8,  # Small step size for gradient approximation
    }

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, constraints=cons, options=first_opt_options)
    v = res.x if res.success else v0
    
    # 1. Phase 1: Asymmetric geometric hash and adaptive stochastic reconfiguration (SOTA-inspired)
    # Perturb with an advanced stochastic hash incorporating spatial density metrics and dynamic scaling
    
    # Extract and analyze current state: centers and radii from current solution
    current_centers = np.column_stack([v[0::3], v[1::3]])
    current_radii = v[2::3]
    
    # Compute spatial metrics to drive adaptive hashing
    dx = current_centers[:, np.newaxis, 0] - current_centers[np.newaxis, :, 0]
    dy = current_centers[:, np.newaxis, 1] - current_centers[np.newaxis, :, 1]
    distances = np.sqrt(dx**2 + dy**2)
    
    # Get min distances in each row for spatial density metrics
    row_density = []
    for row_idx in range(rows):
        # Group circles by row for density calculation
        row_indices = [i for i in range(n) if i // cols == row_idx]
        row_centers = current_centers[row_indices]
        if row_centers.shape[0] == 0:
            row_density.append(0.0)
        else:
            dx_row = row_centers[:, 0] - row_centers[:, 0].mean()
            dy_row = row_centers[:, 1] - row_centers[:, 1].mean()
            # Spatial density metric: average distance to center of mass
            row_dists = np.sqrt(dx_row**2 + dy_row**2)
            avg_row_density = np.mean(row_dists)
            row_density.append(avg_row_density)
    
    # Compute row-wise spatial distortion coefficient for asymmetric hash
    max_density = np.max(row_density) if len(row_density) > 0 else 1.0
    row_density_coeff = row_density / max_density  # Normalize for distortion scaling
    
    # Generate advanced spatial hash with density-based distortion and adaptive scaling
    spatial_hash = np.random.rand(n, 2) * 0.06 * (1 + 0.1 * (row_density_coeff ** 0.75))
    perturbed_v = v.copy()
    for i in range(n):
        perturbed_v[3*i] += np.clip(spatial_hash[i, 0], -0.02, 0.02) * (1 + 0.5 * current_radii[i] / max(current_radii))
        perturbed_v[3*i+1] += np.clip(spatial_hash[i, 1], -0.02, 0.02) * (1 + 0.5 * current_radii[i] / max(current_radii))
    
    # Second optimization on perturbed vector after density-aware spatial reconfiguration
    second_opt_options = {
        "maxiter": 400,  # Reduced from 300 for faster convergence on refined state
        "ftol": 1e-12, 
        "gtol": 1e-12, 
        "eps": 1e-8, 
        "jac": "2-point",  # Use more accurate Jacobian estimation for gradient issues
    }

    res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds, constraints=cons, options=second_opt_options)
    v = res.x if res.success else v
    
    # 2. Phase 2: Targeted expansion with adaptive density-aware expansion factor
    # Calculate distance matrix with optimized vector operations
    dx = np.expand_dims(current_centers[:, 0], axis=1) - np.expand_dims(current_centers[:, 0], axis=0)
    dy = np.expand_dims(current_centers[:, 1], axis=1) - np.expand_dims(current_centers[:, 1], axis=0)
    dists = np.sqrt(dx**2 + dy**2)
    
    # Mask out diagonal (self distances)
    dists = np.ma.masked_where(dists == 0, dists)
    # Compute minimum distance per circle
    min_distances = np.ma.mean(dists, axis=1)
    # Identify least constrained circle based on min distance
    isolated_idx = np.argmin(min_distances)
    
    # Compute expansion factor using density-aware coefficient
    base_expansion = 0.006
    avg_radius = np.mean(current_radii)
    expansion_multiplier = 1.0 + 0.5 * (current_radii[isolated_idx] / avg_radius)
    expansion = base_expansion * expansion_multiplier  # Adaptive expansion based on isolation factor
    
    # Create new radii configuration with targeted expansion
    new_radii = current_radii.copy()
    new_radii[isolated_idx] += expansion * 1.0  # Slight over-expansion to push optimization
    # Distribute expansion to other circles, with density-aware scaling
    for i in range(n):
        if i != isolated_idx:
            # Density-based expansion coefficient: closer circles get more expansion
            radius_i = current_radii[i]
            base = 1.0 + 0.1 * (radius_i / avg_radius)
            # Apply expansion with a small random variation for exploration
            expansion_i = expansion * base * (1.0 + 0.05 * np.random.rand())
            new_radii[i] += expansion_i
    
    # Construct the new decision vector
    v_new = v.copy()
    v_new[2::3] = new_radii
    
    # Third optimization to incorporate the expanded radii
    third_opt_options = {
        "maxiter": 500,
        "ftol": 1e-12,
        "gtol": 1e-12,
        "eps": 1e-8,
        "jac": "2-point",
        "bounds": bounds,
    }
    res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds, constraints=cons, options=third_opt_options)
    v = res.x if res.success else v

    # Final cleanup and validation
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # Clip to prevent NaN or invalid radii

    # Re-verify for any potential edge case
    if np.any(radii <= 0) or np.any(np.isnan(radii)) or np.any(np.isnan(centers)):
        # Fallback to initial v0 if something is wrong
        centers = np.column_stack([v0[0::3], v0[1::3]])
        radii = v0[2::3]
    
    return centers, radii, float(radii.sum())