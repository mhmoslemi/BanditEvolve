import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with adaptive geometric clustering and dynamic staggering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Use adaptive base for staggered grid with dynamic scaling
        col_scale = 1.0 / cols
        row_scale = 1.0 / rows
        x_center = col_scale * (col + 0.5)
        y_center = row_scale * (row + 0.5)
        
        # Dynamic perturbation based on spatial density and row parity
        row_perturbation = 0.0 if row % 2 == 0 else 0.5 * col_scale
        x = x_center + np.random.uniform(-0.06, 0.06) + row_perturbation
        y = y_center + np.random.uniform(-0.06, 0.06)
        
        # Ensure bounds are respected
        if x < 0 or x > 1 or y < 0 or y > 1:
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
        
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

    # Vectorized constraints for boundaries with optimized lambdas
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

    # Vectorized overlap constraints with explicit closure handling
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with default parameters to avoid closure capture issues
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with high precision and adaptive stopping criteria
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-11})
    
    # Apply non-local reconfiguration via geometric hashing and spatial reordering
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create geometric hash with adaptive distribution
        hash_scale = 0.02
        spatial_hash = np.random.rand(n, 2) * hash_scale
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += spatial_hash[i, 0]
            new_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with geometric hashing
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-11})
        
        # Recalculate radii and centers after reconfiguration
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute adjacency matrix to identify topological structure
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        adjacency = np.zeros((n, n), dtype=bool)
        adjacency[dists < 0.5 * (radii + radii[:, np.newaxis]) + 1e-8] = True
        
        # Find least constrained circle with adjacency-based metric
        min_adjs = np.sum(adjacency, axis=1)
        least_constrained_idx = np.argmin(min_adjs)

        # Apply controlled radius expansion with topological reordering
        target_total_sum = np.sum(radii) + 0.007
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
        
        # Create expansion vector with priority on least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2
        
        # Apply expansion to all other circles with spatial consideration
        for i in range(n):
            if i != least_constrained_idx:
                # Compute influence based on adjacency
                influence = np.sum(adjacency[i, :]) / (n - 1)
                expansion = expansion_factor * (1.0 + 0.1 * np.random.rand()) * influence
                new_radii[i] += expansion
        
        # Validate and refine expansion
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
        
        # Update decision vector with refined expansion
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization to refine placement
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())