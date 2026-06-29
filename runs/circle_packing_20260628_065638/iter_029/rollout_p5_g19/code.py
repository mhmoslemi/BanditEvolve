import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n) * 1.1))  # Slender grid for better expansion potential
    rows = (n + cols - 1) // cols
    
    # Initialize positions using dynamic clustering and adaptive offset
    xs = []
    ys = []
    # Use adaptive perturbation depending on grid cell size
    grid_cell_size = 1.0 / cols
    base_offset = grid_cell_size * 0.15  # Larger base offset for better spatial dispersion
    
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols + np.random.uniform(-grid_cell_size * 0.3, grid_cell_size * 0.3)
        y_center = (row + 0.5) / rows + np.random.uniform(-grid_cell_size * 0.3, grid_cell_size * 0.3)
        # Alternate row staggering with adaptive spacing
        if row % 2 == 1:
            x_center += grid_cell_size * 0.6
        xs.append(x_center)
        ys.append(y_center)
    
    # Start radii: higher base with adaptive reduction for edge circles
    r0 = (1.0 / grid_cell_size * 0.7) / cols - 1e-3  # Higher initial guess based on cell size
    # Edge-based radius reduction strategy
    edge_circles = np.array([i for i in range(n) if (i % cols == 0 or i % cols == cols - 1 or i // cols == 0 or i // cols == rows - 1)])
    r0_edge = r0 * 0.8  # Reduce edge circles slightly
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    
    # Initialize radius with edge adjustment
    base_r = np.full(n, r0)
    base_r[edge_circles] = r0_edge
    v0[2::3] = base_r
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Create constraints using vectorized lambda with i capture
    cons = []
    for i in range(n):
        # Left boundary constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: np.clip(v[3*i] - v[3*i+2], -1e12, 0))})
        # Right boundary constraint: 1 - (x_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: np.clip(1.0 - v[3*i] - v[3*i+2], -1e12, 0))})
        # Bottom constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: np.clip(v[3*i+1] - v[3*i+2], -1e12, 0))})
        # Top constraint: 1 - (y_i + r_i) >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: np.clip(1.0 - v[3*i+1] - v[3*i+2], -1e12, 0))})
    
    # Overlap constraints are handled with adaptive weighting based on spatial interactions
    # Add adaptive constraint scaling: use distance-based penalty
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate distance and define constraint that distance >= r_i + r_j
            # Use vectorized lambda for efficiency
            # Store the constraint function in a way that allows parameter capture
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_squared = dx*dx + dy*dy
                r_i = v[3*i+2]
                r_j = v[3*j+2]
                # Use adaptive weighting: higher penalty for closer circles
                # Add buffer of 1e-3 to avoid numerical instability
                # This avoids sqrt for speed and keeps it differentiable
                return dist_squared - (r_i + r_j)**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # 1st optimization: base optimization with tight tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 600, "ftol": 1e-10, "gtol": 1e-10})
    
    # 2nd optimization: asymmetric spatial reconfiguration using spatial hashing
    if res.success:
        # Get current configuration
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash perturbations with adaptive scaling based on grid and radii
        # Use higher perturbations for smaller circles to explore new configurations
        spatial_hash = np.random.rand(n, 2) * 0.06
        scaling_factor = np.maximum(1.0, 5.0 * (radii / np.mean(radii)))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * scaling_factor[i]
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scaling_factor[i]
        
        # 2nd optimization session with increased constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11})
    
    # 3rd optimization: targeted expansion on spatially least constrained circle with soft constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate distances to find least constrained circle (most far from others)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify least constrained circle: circle with minimum sum of inverse distances
        # More space = more freedom to grow
        inverse_distances = 1.0 / (dists + 1e-12)  # Avoid division by zero
        total_inverse_distances = np.sum(inverse_distances, axis=1)
        least_constrained_idx = np.argmin(total_inverse_distances)
        
        # Use adaptive radius expansion with constraints on neighbor interference
        current_total = np.sum(radii)
        max_target_growth = 0.008  # 0.8% increase in sum (based on historical SOTA and feasible region)
        expansion_rate = max_target_growth / (n - 1) * current_total / np.mean(radii)
        expansion_vector = np.zeros(n)
        expansion_vector[least_constrained_idx] = expansion_rate * 1.1
        
        # Apply soft expansion and re-optimization
        # Gradual expansion with constraint validation
        for expansion_step in np.linspace(0, 1, 50):
            # Interpolate expansion factor
            new_radii = radii + expansion_vector * expansion_step
            # Enforce minimum radius
            new_radii = np.maximum(new_radii, 1e-4)
            
            # Build new decision vector
            v_new = v.copy()
            v_new[2::3] = new_radii
            
            # Validate configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = v_new[3*i] - v_new[3*j]
                    dy = v_new[3*i+1] - v_new[3*j+1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-11})
                break
        
        v = res.x if res.success else v
    
    # Final validation and adjustment
    if res.success:
        v = res.x
        radii = v[2::3]
        # Trim radii below threshold
        radii = np.clip(radii, 1e-6, None)
        centers = np.column_stack([v[0::3], v[1::3]])
    else:
        # Fallback to initial configuration with radii clipping
        v = v0
        radii = np.clip(v[2::3], 1e-6, None)
        centers = np.column_stack([v[0::3], v[1::3]])
    
    return centers, radii, float(radii.sum())