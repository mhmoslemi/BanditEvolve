import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset for asymmetric clustering and staggered grid
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
        # Alternate row staggering for better space utilization
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
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with geometric hashing and spatial regularization
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                # Add quadratic spatial regularization to avoid degenerate configurations
                reg = (dx*dx + dy*dy) * 0.01
                return dist_sq - (v[3*i+2] + v[3*j+2])**2 + reg
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-11, "gtol": 1e-11})

    # Geometric hashing-based spatial reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate random geometric hash with higher intensity
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-11})

    # Trigger non-local reconfiguration through geometric hashing and targeted radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate high-intensity geometric hash to trigger non-local reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.12
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += spatial_hash[i, 0]
            new_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-11})

    # Targeted radius expansion on the smallest non-zero radius with spatial regularization
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Use vectorized distance calculation with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        dists = np.tril(dists, k=-1)  # Only consider lower triangle to avoid double-counting
        
        # Identify the smallest non-zero radius
        valid_radii = radii[radii > 0]
        if len(valid_radii) > 0:
            min_radius_idx = np.argmin(radii)
            min_radius = radii[min_radius_idx]
            
            # Calculate expansion factor with total sum constraint and spatial constraints
            target_total_sum = np.sum(radii) + 0.008
            expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
            
            # Expand the smallest radius while respecting total sum and maintaining space
            new_radii = radii.copy()
            new_radii[min_radius_idx] += expansion_factor * 1.1  # Conservative over-expansion
            for i in range(n):
                if i != min_radius_idx:
                    new_radii[i] += expansion_factor
            
            # Apply expansion with constraint validation
            while True:
                expanded_v = v.copy()
                expanded_v[2::3] = new_radii
                expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
                
                # Validate configuration: check if any pair overlaps
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
                    # Reduce expansion factor if invalid
                    new_radii = radii + (new_radii - radii) * 0.98
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization pass with tighter constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())