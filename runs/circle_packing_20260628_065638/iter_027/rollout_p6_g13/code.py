import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with refined geometric clustering and dynamic row spacing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Row center with dynamic spacing
        base_y = (row + 0.5) / rows
        base_y += 0.02 * row
        base_y = np.clip(base_y, 0.05, 0.95)
        x_center = (col + 0.5) / cols
        y_center = base_y
        # Add randomized perturbation with decreasing variance for higher density
        x_pert = np.random.uniform(-0.05, 0.05) if col < cols // 2 else np.random.uniform(-0.03, 0.03)
        y_pert = np.random.uniform(-0.04, 0.04) if row < rows // 2 else np.random.uniform(-0.02, 0.02)
        x = x_center + x_pert
        y = y_center + y_pert
        # Apply staggered grid based on row parity
        if row % 2 == 1:
            x += 0.36 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.36 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with optimized closure
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tighter tolerances and extended iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "gtol": 1e-12})

    # Asymmetric reconfiguration: stochastic spatial hashing with adaptive perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash with adaptive scaling
        spatial_hash = np.random.rand(n, 2) * 0.04
        # Perturb by radius proportion to enhance sensitivity for smaller circles
        for i in range(n):
            if radii[i] < 0.05:
                spatial_hash[i] *= 0.5
            elif radii[i] < 0.1:
                spatial_hash[i] *= 0.75
            perturbed_v = v.copy()
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-12})

    # Targeted expansion of least constrained circle with gradient-aware heuristics
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify least constrained circle: maximize minimum distance to other circles
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion based on gradient information and proximity metrics
        gradient = np.gradient(radii)
        expansion_factor = 0.005 * (np.std(gradient) / np.mean(gradient) + 1)
        
        # Generate dynamic expansion weights for nearby circles
        nearby_indices = np.argsort(min_dists)  # Least constrained first
        expansion_weights = np.zeros(n)
        expansion_weights[least_constrained_idx] = 1.2
        for i in range(1, 5):
            if nearby_indices[i] != least_constrained_idx:
                expansion_weights[nearby_indices[i]] = 0.8

        # Create new radii with strategic expansion
        new_radii = radii.copy()
        for i in range(n):
            new_radii[i] += expansion_factor * expansion_weights[i] * (1.0 + 0.1 * np.random.rand())
        
        # Validate expansion with soft constraints
        iterations = 0
        while iterations < 3:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            valid = True
            
            # Validate distances between circles using vectorized operations
            dists = np.sqrt((expanded_centers[:, np.newaxis, 0] - expanded_centers[np.newaxis, :, 0])**2 +
                            (expanded_centers[:, np.newaxis, 1] - expanded_centers[np.newaxis, :, 1])**2)
            overlaps = np.any(dists < new_radii[:, np.newaxis] + new_radii[np.newaxis, :] - 1e-12, axis=1)
            if np.any(overlaps):
                valid = False
                
            if valid:
                break
            else:
                # Adjust expansion factor for non-overlapping
                new_radii = radii + (new_radii - radii) * 0.95
                iterations += 1
        
        # Update decision vector with refined radii
        v_new = v.copy()
        v_new[2::3] = np.clip(new_radii, 1e-6, 0.5)
        
        # Final optimization with tight constraints and high precision
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())