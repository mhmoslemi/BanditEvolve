import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using dynamic geometric hashing with spatial-aware density control
    # First, compute primary grid positions with dynamic spacing
    grid_x = np.linspace(0.0, 1.0, cols + 1)
    grid_y = np.linspace(0.0, 1.0, rows + 1)
    
    # Define anisotropic weighting based on grid spacing (x is more densely packed)
    x_weight = 0.75
    y_weight = 1.0 - x_weight
    # Generate grid indices with a bias to avoid central clustering
    indices = np.arange(n)
    shuffled_indices = np.random.permutation(n)
    
    xs = []
    ys = []
    for idx in shuffled_indices:
        row = idx // cols
        col = idx % cols
        
        # Use non-uniform grid spacing for better boundary utilization
        x_center = (grid_x[col] + grid_x[col+1]) * 0.5 * (1.0 - 0.3 * (row % 2))
        y_center = (grid_y[row] + grid_y[row+1]) * 0.5 * (1.0 - 0.3 * (col % 2))
        
        # Apply spatially adaptive jittering to avoid regular patterns
        x_jitter = np.random.uniform(-0.005, 0.005) * (1.0 + (row * col) / (n ** 0.5))
        y_jitter = np.random.uniform(-0.005, 0.005) * (1.0 + (row * col) / (n ** 0.5))
        
        # Alternate vertical offset for staggered grid with adaptive spacing
        if row % 3 == 1:
            y_center += 0.04 / (rows + 1)
        
        # Add jitter while maintaining proximity to grid
        x = x_center + x_jitter
        y = y_center + y_jitter
        
        # Ensure positions are within bounds with margin of safety
        x = np.clip(x, 0.0 + 1e-8, 1.0 - 1e-8)
        y = np.clip(y, 0.0 + 1e-8, 1.0 - 1e-8)
        xs.append(x)
        ys.append(y)
    
    # Initialize radii using grid spacing and a soft radius distribution
    max_radius = 0.35 / cols
    r0 = max_radius - 1e-3
    radius_distribution = np.random.exponential(1.0, size=n)
    radius_distribution /= np.sum(radius_distribution)
    radii_initial = r0 + radius_distribution * (max_radius - r0) * 0.9
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = radii_initial
    
    # Ensure bounds are consistent with v0 length
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    # Optimization function to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraints: boundary and non-overlap
    cons = []
    # Add boundary constraints for all circles
    for i in range(n):
        # Left margin
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right margin
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom margin
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top margin
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Add mutual non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Use a dynamic margin based on grid spacing for better convergence
            delta = 0.02 * (cols - col) * (rows - row) / (cols * rows)
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j, _delta=delta: 
                                  (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                  - (v[3*i+2] + v[3*j+2] + _delta)**2)})

    # Main optimization run with enhanced configuration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-10, "disp": False})
    
    # Post-optimization spatial restructuring via hierarchical perturbation
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute hierarchical spatial hash for multi-level constraint perturbation
        # First, get influence matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Create non-overlap-aware spatial hierarchy
        min_dists = np.min(dists, axis=1)
        # Compute a hybrid influence metric: inverse of min_dist + proportional to radius
        influence = 1.0 / (min_dists + 1e-6) + radii * 0.5
        # Normalize the influence
        influence_norm = influence / np.max(influence)
        
        # Create spatial perturbation based on influence hierarchy
        spatial_perturbation = np.random.rand(n, 2) * 0.01
        # Apply more aggressive perturbation to highly influential nodes
        aggressive_perturbation_mask = influence_norm > 0.8
        spatial_perturbation[aggressive_perturbation_mask] *= 2.5
        
        # Apply perturbation to current decision vector
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_perturbation[i, 0] * (radii[i] / np.mean(radii) ** 0.6)
            perturbed_v[3*i+1] += spatial_perturbation[i, 1] * (radii[i] / np.mean(radii) ** 0.6)
        
        # Re-optimization with spatially perturbed vector
        # Add explicit constraint regularization
        extra_cons = []
        for i in range(n):
            extra_cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
            extra_cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
            extra_cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
            extra_cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        
        # Run fine-tuning phase with enhanced constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons + extra_cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-10})
    
    # Apply adaptive geometric perturbation with radius-based expansion
    # Identify the most constrained circle using spatial influence
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute distances matrix again
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute geometric hierarchy based on distance to nearest neighbor
        min_dists = np.min(dists, axis=1)
        # Compute a hybrid metric: min distance + proportional to radius
        hierarchy_metric = min_dists + radii * 0.4
        
        # Find the circle with the highest hierarchy metric (most constrained)
        most_constrained_idx = np.argmax(hierarchy_metric)
        most_constrained_radius = radii[most_constrained_idx]
        
        # Calculate growth with radius-based expansion coefficient
        expansion_coefficient = 1.2 if most_constrained_radius < 0.06 else 1.0
        expansion_amount = (0.006) * exp(-1.5 * (most_constrained_radius - 0.04)) * expansion_coefficient
        
        # Create expansion vector with controlled expansion
        new_radii = radii.copy()
        new_radii[most_constrained_idx] += expansion_amount  # Targeted expansion
        
        # Add moderate stochastic expansion to other circles with spatial awareness
        for i in range(n):
            if i != most_constrained_idx:
                # Use radii-based growth rate and spatial influence
                growth_rate = 0.7 * (1.0 - (i / n))
                influence_factor = (1.0 + (hierarchy_metric[i] - hierarchy_metric[most_constrained_idx]) * 0.3)
                new_radii[i] += growth_rate * influence_factor * 0.002
        
        # Apply expansion with constraint validation using vectorized approach
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Compute all pairwise distances
            dx = expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0]
            dy = expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Validate using efficient vectorized operations
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    if dists[i, j] < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly
                shrink_amount = 0.95
                new_radii = radii + (new_radii - radii) * shrink_amount
        
        # Final optimization with updated radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-11, "eps": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())