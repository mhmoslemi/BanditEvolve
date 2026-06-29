import numpy as np

def run_packing():
    n = 26
    cols = 6
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and multi-pass tiling
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering with dynamic range
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        # Staggered grid with dynamic row offset
        if row % 2 == 1:
            x += (0.8 / cols) * (0.9 - row/rows)
        # Add small random perturbation to avoid symmetry
        x += np.random.uniform(-0.015, 0.015)
        y += np.random.uniform(-0.015, 0.015)
        xs.append(x)
        ys.append(y)
    
    # Initial radius calculation with enhanced spatial density awareness and dynamic scaling
    base_radius = 0.395 / cols - 5e-3
    r0 = base_radius * (1.0 + np.random.uniform(-0.015, 0.015))  # Small random variation
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    # Ensure the bounds list has 3*n entries for the vector of length 3n
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries using lambda with captured i
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Vectorized overlap constraints with spatial hashing and adaptive resolution
    for i in range(n):
        for j in range(i + 1, n):
            # Introduce dynamic resolution adjustment: higher for closely spaced circles
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_radius = min(v[3*i+2], v[3*j+2])
                # Use dynamic resolution: lower resolution for more distant circles
                resolution = max(1.0, np.sqrt(dist_sq) / (min_radius * 2))
                return dist_sq - (v[3*i+2] + v[3*j+2])**2 - 1e-12 * resolution**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with high resolution and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 5e-12, "eps": 1e-10})

    # Advanced asymmetric reconfiguration using directional hashing and nonlocal spatial reordering
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash for perturbation with adaptive scaling and density awareness
        spatial_hash = np.random.rand(n, 2) * 0.06
        # Generate directional hash for reordering with density-based weighting
        directional_hash = np.random.rand(n, 2) * (0.1 + 0.05 * (radii / np.mean(radii)))
        perturbed_v = v.copy()
        
        # Apply directional perturbations with spatial and radii-based weighting
        for i in range(n):
            # Spatial perturbation
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
            # Directional perturbation for ordering based on spatial hashing
            perturbed_v[3*i+2] += directional_hash[i, 0] * (0.005 * (radii[i] / np.mean(radii)))
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 450, "ftol": 5e-12, "eps": 1e-10})

    # Global reordering of radii: identify the spatially isolated circle with most expansion potential
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation using broadcasting with resolution-aware scaling
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Create resolution tensor for nonlocal reconfiguration: higher for distant circles
        resolution_tensor = np.sqrt(dists) / (radii[:, np.newaxis] + radii[np.newaxis, :])
        resolution_tensor = np.max(resolution_tensor, axis=1)
        
        # Find least constrained circle by maximizing minimum distance adjusted by resolution
        min_dists = np.min(dists, axis=1)
        adjusted_min_dists = min_dists * (1.0 + 0.1 * resolution_tensor)
        least_constrained_idx = np.argmax(adjusted_min_dists)
        
        # Calculate growth based on current total and potential for expansion
        current_total = np.sum(radii)
        target_growth = 0.011
        base_growth_ratio = target_growth / (n - 1)
        expansion_factor = base_growth_ratio * (current_total / np.sum(radii))
        direction_factor = np.random.rand(n, 2) * 0.2
        
        # Apply growth with nonlocal spatial constraints and directional bias
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                # Directional expansion based on spatial hashing and neighbor relationships
                adj_weight = np.linalg.norm(centers[least_constrained_idx] - centers[i])
                if adj_weight < 0.1:
                    expansion = expansion_factor * 1.5 * (1.0 + direction_factor[i, 0] * 0.3)
                else:
                    expansion = expansion_factor * 1.0 * (1.0 + direction_factor[i, 1] * 0.4)
                new_radii[i] += expansion * (1.0 + np.random.uniform(-0.01, 0.01))
        
        # Apply expansion with constraint validation using resolution-aware constraints
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration with resolution-aware check
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_radius = min(expanded_v[3*i+2], expanded_v[3*j+2])
                    # Use resolution-aware constraint with spatial and radii awareness
                    resolution = np.sqrt(dist) / (min_radius * 2)
                    if dist < expanded_v[3*i+2] + expanded_v[3*j+2] - 1e-12 * resolution**2:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly with adaptive factor based on density
                new_radii = radii + (new_radii - radii) * (0.95 - 0.01 * np.random.rand())
        
        # Update decision vector with reconfigured positions and expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 450, "ftol": 5e-12, "eps": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())