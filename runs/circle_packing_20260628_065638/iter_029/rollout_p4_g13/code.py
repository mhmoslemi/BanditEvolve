import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initial randomized perturbation on the staggered grid with increased diversity
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base spacing with 10x spatial diversity for cluster separation
        col_base = (col + 0.5) / cols
        row_base = (row + 0.5) / rows
        x_center = col_base + np.random.uniform(-0.03, 0.03) * 1.2
        y_center = row_base + np.random.uniform(-0.03, 0.03) * 1.2
        
        # Staggered offset with row-dependent spatial scaling to reduce congestion
        if row % 2 == 1:
            x_center += np.random.uniform(-0.12, 0.12) * (0.5 / cols)
        
        # Ensure within square without overlap pre-check
        xs.append(np.clip(x_center, 1e-8, 1 - 1e-8))
        ys.append(np.clip(y_center, 1e-8, 1 - 1e-8))
    
    r0 = 0.25 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)]  # Tighter min radius for more precision

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Optimized boundary constraints with lambda closures with i captured
    cons = []
    for i in range(n):
        # Left boundary + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right boundary - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom boundary + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top boundary - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})

    # Optimized overlap constraints with precomputed distance matrix for faster checking
    overlap_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            # Use closure capture for dynamic constraint definition
            overlap_cons.append({"type": "ineq",
                                 "fun": (lambda v, i=i, j=j: 
                                         (v[3*i] - v[3*j])**2 + 
                                         (v[3*i+1] - v[3*j+1])**2 - 
                                         (v[3*i+2] + v[3*j+2])**2)})

    cons.extend(overlap_cons)

    # Initialize optimization with SLSQP with enhanced parameters
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1200, "ftol": 1e-10, "eps": 1e-8})

    # Spatial reconfiguration with adaptive geometric hashing to escape local optima
    if res.success:
        v = res.x
        # Generate spatial hash with adaptive radius-dependent perturbation
        radii = v[2::3]
        mean_radius = np.mean(radii)
        hash_map = np.random.rand(n, 2) * (0.1 * (radii / mean_radius))
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_map[i, 0] * mean_radius * 1.2
            perturbed_v[3*i+1] += hash_map[i, 1] * mean_radius * 1.2
            # Ensure perturbation doesn't push out of boundaries
            perturbed_v[3*i] = np.clip(perturbed_v[3*i], 1e-8, 1 - 1e-8)
            perturbed_v[3*i+1] = np.clip(perturbed_v[3*i+1], 1e-8, 1 - 1e-8)
        # Re-evaluate with perturbed parameters but with stricter bounds
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Advanced radius expansion with spatial-aware expansion prioritization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx ** 2 + dy ** 2)
        
        # Calculate minimum distance to others for each circle
        min_dists = np.min(dists, axis=1)
        # Find circle with minimum minimum distance (most constrained)
        least_constrained_idx = np.argmin(min_dists)
        # Find circles with largest distance to neighbors (least constrained)
        max_dists = np.max(dists, axis=1)
        most_unconstrained_idx = np.argmax(max_dists)

        current_total = np.sum(radii)
        # Compute expansion potential based on radius variance
        std_radius = np.std(radii)
        target_growth = 0.0065
        expansion_factor = target_growth / std_radius * 1.2
        # Add a spatial-aware expansion vector
        new_radii = radii.copy()
        
        # Targeted expansion on most un-constrained and least-constrained circles
        new_radii[least_constrained_idx] += expansion_factor * 1.1
        new_radii[most_unconstrained_idx] += expansion_factor * 0.9
        
        # Distribute the rest of the expansion to other circles with adaptive scaling
        for i in range(n):
            if i not in [least_constrained_idx, most_unconstrained_idx]:
                # Use distance to other circles to prioritize expansion
                dist_to_neighbors = np.mean(dists[i, np.arange(n) != i])
                expansion_i = expansion_factor * (0.8 + (1 - dist_to_neighbors / (1 - 1e-8)) * 0.2)
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation
        while True:
            # Apply the expansion
            expanded_v = v.copy()
            expanded_v[2::3] = np.clip(new_radii, 1e-6, 0.5)
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_exp = np.sqrt(dx_exp ** 2 + dy_exp ** 2)
                    if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion if invalid
                new_radii = radii + (new_radii - radii) * 0.98
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        # Re-evaluate with more aggressive optimization
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-9})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # Cap radii at 0.5
    return centers, radii, float(radii.sum())