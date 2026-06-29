import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Random seed to ensure reproducible initialization
    np.random.seed(42)
    
    # Initialize spatial configuration with asymmetric randomized geometric hashing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Generate base grid positions with offset
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add randomized geometric hash for asymmetry and spread
        x_offset = np.random.uniform(-0.09, 0.09)
        y_offset = np.random.uniform(-0.09, 0.09)
        # Alternate row shift to create staggered grid and break symmetry
        if row % 2 == 1:
            x_offset += 0.5 / cols * np.random.choice([-1, 1]) * 0.7
        x = x_center + x_offset
        y = y_center + y_offset
        xs.append(x)
        ys.append(y)
    
    r0 = 0.37 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
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
    
    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "maxls": 100})
    
    # First-stage geometric reconfiguration with spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate geometric hash for spatial reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Apply perturbation to spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "maxls": 100})
    
    # Second-stage targeted radius expansion on smallest non-overlapping circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Efficient pairwise distance calculation using broadcasting to identify
        # isolated circle with smallest non-zero radius and highest isolation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        valid_dists = np.where(dists > 1e-12, dists, np.inf)
        
        # Calculate minimum distance to others
        min_dist = np.min(valid_dists, axis=1)
        
        # Filter to remove circles with zero radius (possible due to constraints)
        valid_indices = np.where(radii > 1e-6)[0]
        valid_radii = radii[valid_indices]
        valid_centers = centers[valid_indices]
        valid_dists = dists[valid_indices][:, valid_indices]
        valid_min_dist = np.min(valid_dists, axis=1)
        
        # Identify most isolated circle among valid radii
        isolated_idx = np.argmax(valid_min_dist)
        isolated_circle_idx = valid_indices[isolated_idx]
        
        # Calculate expansion factor based on minimum spacing
        min_spacing = np.min(valid_min_dist)
        expansion_factor = (min_spacing - 2 * radii[isolated_circle_idx]) / (n - 1)
        expansion_factor *= 1.1  # Introduce slight over-expansion to trigger reconfiguration
        
        # Apply radius expansion with soft constraints
        new_radii = radii.copy()
        new_radii[isolated_circle_idx] += expansion_factor
        for i in range(n):
            if i != isolated_circle_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.05 * np.random.rand())
        
        # Validate new radii against non-overlap constraints and update decision vector
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
        valid = True
        
        # Validate non-overlap with spatial constraints
        for i in range(n):
            r = new_radii[i]
            if r < 1e-6:
                valid = False
                break
            for j in range(i + 1, n):
                dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < r + new_radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "maxls": 100})
        else:
            # If validation fails, apply more conservative expansion
            new_radii = radii.copy()
            new_radii[isolated_circle_idx] += expansion_factor * 0.8
            for i in range(n):
                if i != isolated_circle_idx:
                    new_radii[i] += expansion_factor * 0.6 * (1.0 + 0.05 * np.random.rand())
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-11, "maxls": 100})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())