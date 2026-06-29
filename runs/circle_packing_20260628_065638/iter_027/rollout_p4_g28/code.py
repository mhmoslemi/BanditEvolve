import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering, staggered grid, and spatial randomness
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x_offset = np.random.uniform(-0.08, 0.08)
        y_offset = np.random.uniform(-0.08, 0.08)
        # Adjust for alternating row staggering if applicable
        offset = (0.5 / cols) * (1 if row % 2 == 1 else 0.0)
        x = x_center + x_offset
        y = y_center + y_offset
        # Apply small spatial perturbation for diversity
        x += np.random.uniform(-0.01, 0.01)
        y += np.random.uniform(-0.01, 0.01)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using bound-aware lambdas
    cons = []
    for i in range(n):
        # Left + radius <= 1 (bound constraint)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0 (bound constraint)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1 (bound constraint)
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0 (bound constraint)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints using lambda with explicit i and j capture
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-11})
    
    # Radical reconfiguration via randomized geometric tiling
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate random geometric tiling for reconfiguration
        random_tiling = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            # Spatial reconfiguration based on radius and randomness
            perturbed_v[3*i] += random_tiling[i, 0] * (radii[i] / np.min(radii))
            perturbed_v[3*i+1] += random_tiling[i, 1] * (radii[i] / np.min(radii))
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    # Targeted radius expansion using minimal constraint circles and dynamic sum expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by minimizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        least_constrained_radius = radii[least_constrained_idx]
        
        # Calculate growth based on current total sum and potential for expansion
        current_total = np.sum(radii)
        total_sum_target = 2.645  # Dynamic target to push beyond current best
        expansion_factor_base = (total_sum_target - current_total) / (n - 1)
        
        # Create expansion vector with targeted expansion on least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor_base * 1.3  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor_base * (1.0 + 0.15 * np.random.rand())  # Stochastic expansion
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation and dynamic adjustment
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})

    # Final configuration
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())