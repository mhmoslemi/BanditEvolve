import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols

    # Initialize positions using a refined geometric tiling with enhanced stochasticity
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add more nuanced randomized spatial displacement to prevent clustering and enable better optimization
        x = x_center + np.random.uniform(-0.03, 0.03)
        y = y_center + np.random.uniform(-0.03, 0.03)
        # Shift alternate rows to create staggered grid for even spacing
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)

    # Set initial radius as per tiling geometry with higher potential for expansion
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds for all 3n variables (3 per circle)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundary conditions
    cons = []
    for i in range(n):
        # Left: x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right: x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom: y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top: y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized constraints for circle-circle non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j:
                         (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                         - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with increased max iterations and tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "eps": 1e-12})

    # Radical reconfiguration through randomized geometric tiling with adaptive spacing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply randomized geometric perturbation with spatial and radius-based scaling
        spatial_hash = np.random.rand(n, 2) * 0.1  # Higher variance for radical reconfiguration
        radius_factor = 0.01 * (np.max(radii) / np.min(radii)) * 0.7
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        # Re-evaluate with new geometric configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-12})

    # Targeted radius expansion on least constrained circle with soft constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation for optimized performance
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Find least constrained circle with improved metric
        # Metric: max(min distance to others while maintaining minimum distance to wall
        min_dists = np.min(dists, axis=1)
        min_boundary_dist = np.min(np.minimum(centers[:,0] + radii, 1.0 - radii),
                                  np.minimum(centers[:,1] + radii, 1.0 - radii))
        constrained_idx = np.argsort(-min_dists * (1.0 - 0.8 * (np.max(radii) - np.min(radii)) / np.max(radii)))
        least_constrained_idx = constrained_idx[0]  # Least constrained based on combined metric

        # Calculate growth based on current total sum and potential for expansion
        current_total = np.sum(radii)
        target_growth = 0.006  # Fixed absolute expansion target for controlled exploration
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Create expansion vector with directed expansion on least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3  # Slightly over-expanding key circle
        # Additional small expansion to neighbors with spatial-aware adjustment
        for i in range(n):
            if i == least_constrained_idx:
                continue
            # Compute distance to least constrained circle for spatial consideration
            distance = np.sqrt((centers[i, 0] - centers[least_constrained_idx, 0])**2
                              + (centers[i, 1] - centers[least_constrained_idx, 1])**2)
            # Increase expansion for nearby circles
            if distance < (radii[i] + radii[least_constrained_idx])*1.5:
                expansion = expansion_factor * 1.2  # 20% more expansion for neighbors within 150% of sum of radii
                new_radii[i] += expansion
            else:
                expansion = expansion_factor * 0.8  # 20% less expansion for distant circles
                new_radii[i] += expansion
        
        # Apply expansion with constraint validation and adaptive scaling
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
                # If invalid, decrease expansion slightly proportional to constraint violation
                # Compute worst constraint violation
                worst_violation = np.inf
                for i in range(n):
                    for j in range(i+1, n):
                        dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                        dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                        dist = np.sqrt(dx**2 + dy**2)
                        constraint_violation = dist - (new_radii[i] + new_radii[j])
                        if constraint_violation < worst_violation:
                            worst_violation = constraint_violation
                # Proportional reduction in expansion based on constraint severity
                reduction_factor = np.clip(worst_violation + 1e-6, 0.0, 1.0)
                new_radii = radii + (new_radii - radii) * (1.0 - reduction_factor)
                new_radii = np.clip(new_radii, 1e-6, 0.5)
        
        # Update decision vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-12})

    # Final decision vector and results
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)
    return centers, radii, float(radii.sum())