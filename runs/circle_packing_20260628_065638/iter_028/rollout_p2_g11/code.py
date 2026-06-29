import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize positions with randomized geometric clustering and controlled staggered grid
    xs = []
    ys = []
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        # Base grid center positions
        x_center = (col_idx + 0.5) / cols
        y_center = (row_idx + 0.5) / rows
        # Apply randomized offset with tighter bounds
        offset_x = np.random.uniform(-0.04, 0.04)
        offset_y = np.random.uniform(-0.04, 0.04)
        # Add row-specific staggering to improve spacing
        if row_idx % 2 == 1:
            x_center += 0.5 / cols * 0.6  # Less aggressive than original to prevent over-perturbation
        # Ensure centers remain in valid range
        x = max(min(x_center + offset_x, 1 - 1e-7), 0.0 + 1e-7) # Avoid edge issues
        y = max(min(y_center + offset_y, 1 - 1e-7), 0.0 + 1e-7)
        xs.append(x)
        ys.append(y)
    
    # Base radius with adaptive scaling and improved spatial awareness
    base_radius = 0.37 / cols - 1e-3
    r0 = base_radius * np.ones(n)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    # Define bounds that have exactly 3*n dimensions
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with lambda captures for correct i
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with tighter closure capture
    for i in range(n):
        for j in range(i + 1, n):
            # Using lambda with explicit captures for i and j
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with tighter constraints and higher iterations
    initial_res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 2000, "ftol": 1e-11, "eps": 1e-8})
    
    if initial_res.success:
        v = initial_res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply spatial reconfiguration with structured randomness
        # Create spatial noise vector with adaptive scaling
        spatial_noise = np.random.rand(n, 2) * 0.03
        perturbed_v = v.copy()
        for i in range(n):
            # Apply spatial displacement proportionally to radius
            perturbed_v[3*i] += spatial_noise[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_noise[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-evaluate with new spatial configuration
        reconf_res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                              constraints=cons, options={"maxiter": 600, "ftol": 1e-11})
    
    # Apply constrained reconfiguration with targeted expansion: find least constrained circle
    if reconf_res.success:
        v = reconf_res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Use vectorized distance matrix calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with maximum minimum distance
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Use gradient-based radius expansion with controlled scaling
        current_total = np.sum(radii)
        target_growth_percentage = 0.008  # 0.8% increase over current total
        target_total_sum = current_total + target_growth_percentage * current_total
        expansion_factor = (target_total_sum - current_total) / (n - 1)
        
        # Apply expansion with soft stochastic constraints
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.15
        for i in range(n):
            if i != least_constrained_idx:
                # Add random variation in expansion between +0 to +0.2 of the factor
                random_factor = 0.2 * np.random.rand()
                new_radii[i] += expansion_factor * (1 + random_factor)
        
        # Gradient-based optimization of expanded configuration
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate distances with fast numpy check
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion by 5% if invalid; this is a fallback heuristic
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Re-run optimization with expanded radii
        final_res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                            constraints=cons, options={"maxiter": 600, "ftol": 1e-11})
    
    # Final evaluation
    v = final_res.x if final_res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())