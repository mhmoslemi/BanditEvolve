import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))  # More balanced grid than fixed cols=5
    rows = (n + cols - 1) // cols
    
    # Radical reconfiguration: initialize via randomized spatial hashing with non-local distribution
    xs, ys = [], []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Base grid centers
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Add non-local randomized offsets to disrupt spatial regularity
        x_offset = np.random.uniform(-0.1, 0.1) * (np.sqrt(n) / cols)
        y_offset = np.random.uniform(-0.1, 0.1) * (np.sqrt(n) / rows)
        
        # Dynamic staggered grid adjustment based on row parity and distance to edges
        if row % 2 == 1:
            x_center += (0.5 / cols) * (np.random.choice(np.array([1, -1])))
        
        # Apply offsets with soft clipping
        x = np.clip(x_center + x_offset, 0, 1)
        y = np.clip(y_center + y_offset, 0, 1)
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimation using spatial density and randomized adjustments
    r0 = np.clip(0.35 / (cols * rows) - 1e-3 + np.random.uniform(-0.01, 0.01), 1e-4, 0.5)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n length

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with lambda closures
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})

    # Vectorized non-overlap constraints with advanced distance calculation
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx**2 + dy**2
                return dist_sq - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased precision and max iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})
    
    # Radical reconfiguration with adaptive geometric hashing and spatial perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create spatial hash for dynamic configuration reshuffling
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            # Adaptive scaling based on local cluster density
            radius_scale = radii[i] / (np.mean(radii) + 1e-8)
            perturbed_v[3*i] += spatial_hash[i, 0] * radius_scale
            perturbed_v[3*i+1] += spatial_hash[i, 1] * radius_scale
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    # Targeted reconfiguration: spatially aware expansion with geometric reordering
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation for minimum distance analysis
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with least spatial constraint using advanced measure
        # We use (1/min_dist) * (1/(num_neighbors)) as a composite metric
        min_dists = np.min(dists, axis=1)
        neighbor_counts = np.sum(dists < np.sqrt((radii + np.min(radii))**2), axis=1)
        least_constrained_idx = np.argmax(1 / (min_dists[neighbor_counts > 0] + 1e-6))
        
        # Compute expansion factors based on dynamic scaling
        current_total = np.sum(radii)
        max_potential = (np.sqrt(2) - 1) * (1 - np.max(radii))
        expansion_factor = max_potential / (n - 1) * (current_total / np.sum(radii))
        
        # Create expansion vector with adaptive expansion to least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3  # Stronger over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                expansion = expansion_factor * (1.0 + 0.1 * np.random.rand())
                new_radii[i] += expansion * (1.0 - np.random.rand())  # Soft adjustment
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i+1, n):
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
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    # Final validation and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())