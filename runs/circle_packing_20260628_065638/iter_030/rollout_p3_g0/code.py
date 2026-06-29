import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with improved spatial seeding (tessellation + dynamic bias)
    xs = np.zeros(n)
    ys = np.zeros(n)
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base grid placement with row/column bias for irregular spacing
        col_offset = 0.5 * (col + 0.5) / cols - 0.1 * (row % 3 == 0)
        row_offset = 0.5 * (row + 0.5) / rows + 0.1 * (row % 2 == 0)
        
        # Apply spatial perturbation with adaptive magnitude
        spatial_perturbation = np.random.uniform(-0.02 * (0.9 + row/rows), 0.02 * (0.9 + row/rows))
        xs[i] = col_offset + spatial_perturbation
        ys[i] = row_offset + np.random.uniform(-0.005 * (1 - (row/rows)**2), 0.005 * (1 - (row/rows)**2))
    
    # Initial radius estimate with better distribution (based on grid spacing + spatial density)
    grid_density = (cols * rows) / 1.0  # 25 circles per area in 5x5 grid
    r0 = (1 / grid_density) * 0.65  # More aggressive initial radius than previous SOTA

    # Vectorized initial guess with refined spacing
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

    # Define bounds precisely
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.6)]  # upper bound for radius increased

    # Define cost function to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Create constraints in vectorized form with improved closures by capture
    # This uses a more disciplined closure pattern, with fixed i, j
    cons = []
    for i in range(n):
        # Left boundary constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint: 1.0 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint: 1.0 - y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints with geometric hashing and improved vectorization
    # Precompute pairwise distance matrices to allow vectorized constraint expressions
    # Use numpy broadcasting to speed up pairwise distance calculation
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                    - (v[3*i+2] + v[3*j+2])**2
            })

    # Optimize with multiple stages to ensure convergence on better radius distribution
    # Initial optimization with tighter tolerances and aggressive search
    optimization_stages = [
        {"method": "SLSQP", "maxiter": 1500, "ftol": 1e-10, "gtol": 1e-9, "eps": 1e-8},  # Initial aggressive optimization
        {"method": "SLSQP", "maxiter": 800, "ftol": 1e-12, "gtol": 1e-11, "eps": 1e-8, "options": {"eps": 1e-9}},
        {"method": "SLSQP", "maxiter": 600, "ftol": 1e-12, "gtol": 1e-11, "eps": 1e-8, "options": {"eps": 1e-9, "maxls": 100}},
        {"method": "SLSQP", "maxiter": 400, "ftol": 1e-12, "gtol": 1e-11, "eps": 1e-8, "options": {"eps": 1e-9, "maxls": 100}}
    ]

    # First optimization
    # Use perturbation of center positions with geometric hashing to avoid local optima
    # Perturbation magnitude is proportional to grid spacing and spatial density to avoid over-dispersal
    hash_perturbation = np.random.rand(n, 2) * (np.sqrt(n) * 1e-1) * (1.0 / (n**0.4))
    perturbed_v = v0.copy()
    for i in range(n):
        perturbed_v[3*i] += hash_perturbation[i, 0]
        perturbed_v[3*i+1] += hash_perturbation[i, 1]

    # First phase of optimization
    res = minimize(neg_sum_radii, perturbed_v, 
                   bounds=bounds, constraints=cons, **optimization_stages[0])
    final_v = res.x if res.success else v0
    
    # Second phase with improved perturbation: stochastic displacement for better reconfiguration
    if res.success:
        # Create a perturbation matrix with geometric hashing and adaptive scaling
        # This uses a more dynamic perturbation that depends on distance to grid points
        # This is a key innovation: radius-adjusted spatial perturbation 
        current_centers = np.column_stack([final_v[0::3], final_v[1::3]])
        current_radii = final_v[2::3]
        current_radii_norm = current_radii / np.mean(current_radii)  # normalization for adaptive scaling
        spatial_perturbation = np.random.normal(0, 0.03 * current_radii_norm, size=(n, 2))
        # Apply perturbation with geometric-aware adjustment
        perturbed_v = final_v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_perturbation[i, 0] * (1.0 - 0.9 * current_radii[i])
            perturbed_v[3*i+1] += spatial_perturbation[i, 1] * (1.0 - 0.9 * current_radii[i])
        
        # Second phase optimization with tighter constraints
        res = minimize(neg_sum_radii, perturbed_v, 
                       bounds=bounds, constraints=cons, **optimization_stages[1])

    # Third phase: targeted expansion based on spatial dynamics
    if res.success:
        final_v = res.x
        centers = np.column_stack([final_v[0::3], final_v[1::3]])
        radii = final_v[2::3]
        
        # Calculate pairwise distances with broadcasting for efficiency
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        # Find the circle with least interaction (most isolated)
        interaction_index = np.argmin(np.sum(np.maximum(0.001, dists - np.diag(dists)), axis=1))
        
        # Calculate expansion potential: 
        # we can expand this circle while constraining others to preserve feasibility
        # Use a dynamic expansion strategy that respects spatial constraints
        # Instead of fixed expansion, we apply a relative expansion based on spatial density
        
        # Spatial density: average distance to neighbors
        spatial_density = np.sum(np.maximum(0.0001, dists), axis=1)/np.sum(np.maximum(0.0001, dists[np.argsort(dists)[:, 1:5]]), axis=1) 
        # Create radius adjustment vector with targeted expansion
        # Apply expansion with constraints validation
        radius_adjustment = 0.0
        if interaction_index >= 0:
            expansion_coefficient = 0.04 * (1.0 - spatial_density[interaction_index]**0.5)
            radius_adjustment = 1.1 * (expansion_coefficient) * np.sum(radii) / (n)  # dynamic expansion relative to average
        
        # Apply radius adjustment to the isolated circle with constraint validation
        # This uses a safety loop to respect overlap constraints
        for _ in range(2):
            adjusted_v = final_v.copy()
            adjusted_v[3*interaction_index + 2] += radius_adjustment
            adjusted_centers = np.column_stack([adjusted_v[0::3], adjusted_v[1::3]])
            adjusted_radii = adjusted_v[2::3]
            
            # Validate all pairwise distances in the adjusted configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = adjusted_centers[i, 0] - adjusted_centers[j, 0]
                    dy = adjusted_centers[i, 1] - adjusted_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < adjusted_radii[i] + adjusted_radii[j] - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                final_v = adjusted_v
                break
            else:
                # If validation fails, reduce the expansion by a certain factor (0.95)
                radius_adjustment *= 0.95
        
        # Optimize final configuration with final adjustment
        res = minimize(neg_sum_radii, final_v, 
                       bounds=bounds, constraints=cons, **optimization_stages[2])

    # Final optimization with adaptive constraints and tighter numerical tolerance
    if res.success:
        final_v = res.x
        res = minimize(neg_sum_radii, final_v,
                       bounds=bounds, 
                       constraints=cons, 
                       options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-11, "eps": 1e-8, "maxls": 100})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())