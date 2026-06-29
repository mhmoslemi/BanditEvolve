import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized grid-based positions allowing adaptive layout
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_base = (col + 0.2) / cols
        y_base = (row + 0.2) / rows
        # Add a small random offset to avoid grid alignment
        x = x_base + np.random.uniform(-0.03, 0.03)
        y = y_base + np.random.uniform(-0.03, 0.03)
        # Shift alternate rows for staggered effect
        if row % 2 == 1:
            x += 0.5 / cols
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

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with geometric hashing for spatial awareness
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx**2 + dy**2
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with high precision and parallelization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "eps": 1e-12})
    
    # Apply spatial hashing and random perturbation for radical reconfiguration
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Create a spatial hash grid to guide the configuration
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with hash-informed configuration
        res2 = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                        constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-12})
        if res2.success:
            v = res2.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])

    # Identify the circle with smallest radius and apply controlled expansion
    if res.success or res2.success:
        dists = np.zeros((n, n))
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with smallest radius and lowest minimum distance
        min_dists = np.min(dists, axis=1)
        min_radius = np.min(radii)
        small_radii_idx = np.where(radii == min_radius)[0]
        least_constrained_idx = small_radii_idx[np.argmin(min_dists[small_radii_idx])]
        
        # Apply controlled expansion to this circle with soft constraints
        target_total_sum = np.sum(radii) + 0.006
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
        
        # Create new radii with soft enforcement of expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.05 * np.random.rand())
        
        # Apply expansion with constraint validation
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
                new_radii = radii + (new_radii - radii) * 0.96

        # Final update to decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization to refine the reconfigured layout
        res3 = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                        constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-12})
        if res3.success:
            v = res3.x

    v = res.x if res.success else res2.x if res2.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())