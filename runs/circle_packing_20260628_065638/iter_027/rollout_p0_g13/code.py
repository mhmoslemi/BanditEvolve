import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with a hybrid of dense grid and randomized clusters with explicit row staggering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base grid center
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add randomized perturbation
        x = x_center + np.random.uniform(-0.15, 0.15)
        y = y_center + np.random.uniform(-0.15, 0.15)
        # Add staggered row offset
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimate based on grid spacing with density adjustment
    r0 = 0.34 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds correctly aligned with 3*n elements
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3 entries per circle, total 3n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective to maximize sum of radii

    # Create constraints with proper lambda binding (no closures or lambda tricks)
    cons = []
    for i in range(n):
        # Left margin constraint: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right margin constraint: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom margin constraint: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top margin constraint: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    # Overlap constraints between all circle pairs
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i + 1] - v[3*j + 1]
                return dx*dx + dy*dy - (v[3*i + 2] + v[3*j + 2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tightened tolerances and large max iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, 
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-12})

    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Generate geometric hash perturbation based on adaptive grid scaling
        spatial_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))

        # Refine configuration with new spatial arrangement
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds, 
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-12})

    # Topological reordering with constrained radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Vectorized distance calculation for all circle pairs
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Find circle with least constraint (smallest radius & maximal minimum distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(radii)  # Choose smallest radius instead of largest distance
        if np.isnan(least_constrained_idx):  # Fallback if all radii are same
            least_constrained_idx = np.argmin(dists.mean(axis=1))
        
        # Adaptive radius expansion with local validation
        new_radii = radii.copy()
        current_total = np.sum(radii)
        expansion_factor = 0.008  # Base expansion target
        adjustment = 1.05  # Safety factor to handle rounding and optimization tolerance
        
        # Compute expansion based on distance to neighbors
        neighbor_distances = dists[least_constrained_idx, :]
        # Calculate potential available space
        available_space = np.mean(neighbor_distances - 2*radii[least_constrained_idx])
        if available_space > 0:
            # Calculate safe expansion factor based on available space relative to total radii
            expansion_factor = min(0.02 + 0.01 * available_space / np.mean(radii), 0.02)
        
        # Apply expansion with adaptive scaling
        new_radii[least_constrained_idx] = np.clip(radii[least_constrained_idx] + expansion_factor * adjustment, 
                                                  1e-5, 0.5)
        for i in range(n):
            if i != least_constrained_idx:
                # Use distance-based scaling for other circles
                expansion_i = expansion_factor * (dists[i, least_constrained_idx] / radii[least_constrained_idx])
                new_radii[i] = np.clip(radii[i] + expansion_i * adjustment, 1e-5, 0.5)
        
        # Validate and refine new radii with iterative constraint checking
        iter_count = 0
        while iter_count < 2:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate circle positions and overlap
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
                # Reduce expansion slightly for next iteration
                new_radii = radii + (new_radii - radii) * 0.95
                iter_count += 1
        
        # Update decision vector with refined radii
        v_new = v.copy()
        v_new[2::3] = new_radii

        # Final optimization with refined configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds, 
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())