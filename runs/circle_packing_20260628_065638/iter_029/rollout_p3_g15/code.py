import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with adaptive grid + enhanced perturbation grid and 
    # spatial clustering with directional bias to prevent collapse
    xs = []
    ys = []
    base_grid_size = 0.15
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid with directional bias for better spacing
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Apply grid-specific directional clustering (e.g., bias towards center on even rows)
        directional_bias = (row % 4 == 2) * (0.03) * (np.sin(row * np.pi / 4))
        x = x_center + np.random.uniform(-0.08, 0.08) * np.sqrt(radii_base) + directional_bias
        y = y_center + np.random.uniform(-0.08, 0.08) * np.sqrt(radii_base)

        # Apply row-specific stagger to avoid column collapse
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Base radius estimation with adaptive scaling
    radii_base = np.min([0.35 / cols, 0.4 * np.sqrt((n)/cols)])
    # Initial guess for radii with spatial clustering awareness
    r0 = radii_base - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.45)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with explicit indices
    cons = []
    for i in range(n):
        # Left constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2] - 1e-10})
        # Right constraint: 1 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2] - 1e-10})
        # Bottom constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2] - 1e-10})
        # Top constraint: 1 - y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2] - 1e-10})
    
    # Vectorized overlap constraints with optimized Jacobian calculation
    # Precompute a matrix of all pairwise circle positions for direct access
    # This is more efficient than recomputing for each constraint
    # Using list comprehension to generate per-circle constraints for all pairs
    # with explicit lambda capturing of indices
    overlap_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            overlap_cons.append({"type": "ineq",
                                 "fun": lambda v, i=i, j=j: 
                                     (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})
    cons.extend(overlap_cons)
    
    # Initial optimization with enhanced parameters
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-11, "gtol": 1e-11})
    
    # Adaptive geometric hashing reconfiguration with directional bias
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate direction-aware spatial hashing parameters
        # Use radius-aware perturbation scaling
        radius_norm = radii / radii.mean()
        spatial_hash = np.random.rand(n, 2) * 0.15 * radius_norm
        perturbed_v = v.copy()
        for i in range(n):
            # Apply direction-aware perturbation 
            # Add directional bias proportional to radius and row
            row_index = i // cols
            directional_bias = (row_index % 4 == 2) * (0.1) * radii[i]
            perturbed_v[3*i] += spatial_hash[i, 0] + directional_bias
            perturbed_v[3*i+1] += spatial_hash[i, 1] + directional_bias
            # Add row-specific perturbation to prevent column stacking
            if row_index % 2 == 1:
                perturbed_v[3*i] += (row_index * 0.15) / cols * (1.0 - radius_norm[i])
            # Ensure not to exceed bounds
            if perturbed_v[3*i] > 1.0 - 1e-10:
                perturbed_v[3*i] = 1.0 - 1e-10
            if perturbed_v[3*i] < 1e-2:
                perturbed_v[3*i] = 1e-2
            if perturbed_v[3*i+1] > 1.0 - 1e-10:
                perturbed_v[3*i+1] = 1.0 - 1e-10
            if perturbed_v[3*i+1] < 1e-2:
                perturbed_v[3*i+1] = 1e-2
        
        # Re-optimization with enhanced directional bias
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-11})
    
    # Spatial constraint refinement with targeted non-overlapping expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx**2 + dy**2)
                dists[j, i] = dists[i, j]
        
        # Find circles with the smallest and largest distances to others for targeted expansion
        min_dists = np.min(dists, axis=1)
        max_dists = np.max(dists, axis=1)
        
        # Identify the circle with smallest distance (most constrained)
        most_constrained_idx = np.argmin(min_dists)
        # Identify the circle with largest distances to others (least constrained)
        least_constrained_idx = np.argmax(max_dists)
        
        # Compute expansion based on a hybrid approach: target total growth and individual potential
        current_total = np.sum(radii)
        target_total = current_total + 0.007
        # Compute individual expansion potential based on distance to others
        expansion_potential = (dists[least_constrained_idx, :] / np.max(dists, axis=1)[least_constrained_idx]) * radii[least_constrained_idx]
        avg_potential = np.mean(expansion_potential)
        
        # Calculate expansion factor with safety margin
        expansion_factor = (target_total - current_total) / (n - 1) * 0.8
        # Target expansion for most constrained circle
        max_expand = expansion_factor * 0.8
        # Target expansion for least constrained circle
        min_expand = expansion_factor * 1.1
        
        # Construct new radii vector
        new_radii = radii.copy()
        # Expand most constrained circle first
        new_radii[most_constrained_idx] += max_expand * 0.95
        # Expand least constrained circle to fill in the rest
        new_radii[least_constrained_idx] += min_expand * 0.9
        # Distribute remaining expansion to other circles
        remaining = target_total - current_total - (new_radii[most_constrained_idx] - radii[most_constrained_idx]) - (new_radii[least_constrained_idx] - radii[least_constrained_idx])
        for i in range(n):
            if i != most_constrained_idx and i != least_constrained_idx:
                scale = (expansion_potential[i] / np.max(expansion_potential)
                         if np.max(expansion_potential) > 0 else 1.0)
                new_radii[i] += remaining * scale * 0.6
        
        # Apply new radii with constraint validation to prevent invalid configurations
        # Use vectorized pairwise checks to validate feasibility
        # Optimized pairwise distance calculation using broadcasting
        dx_full = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy_full = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dist_full = np.sqrt(dx_full**2 + dy_full**2)
        
        # Apply new radii
        v_expanded = v.copy()
        v_expanded[2::3] = np.clip(new_radii, 1e-4, 0.45)
        
        # Validate configuration
        valid_config = True
        for i in range(n):
            for j in range(i + 1, n):
                if dist_full[i, j] < v_expanded[3*i+2] + v_expanded[3*j+2] - 1e-12:
                    valid_config = False
        if not valid_config:
            # If validation fails, apply conservative expansion
            # Use a gradient descent-like approach to gradually expand
            alpha = 0.98  # Conservative expansion factor
            new_radii = radii * alpha
            new_radii = np.minimum(new_radii, v_expanded[2::3])
            v_expanded = v.copy()
            v_expanded[2::3] = new_radii
        
        # Refine with optimized local solver
        res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())