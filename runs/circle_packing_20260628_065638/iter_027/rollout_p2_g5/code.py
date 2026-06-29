import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize positions with non-uniform geometric tiling and advanced randomness
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Use non-uniform random offset for asymmetric placement
        x = x_center + np.random.uniform(-0.05, 0.05) * (1.0 + np.random.rand() * 0.3)
        y = y_center + np.random.uniform(-0.05, 0.05) * (1.0 + np.random.rand() * 0.3)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols * (1.0 + np.random.rand() * 0.2)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.45 / cols - 3e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Ensure 3*n entries

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with captured parameters and cleaner lambda closure
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

    # Vectorized overlap constraints with efficient lambda closure
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with captured i,j to avoid nested closure issues
            cons.append({"type": "ineq",
                         "fun": lambda v, i=i, j=j: 
                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                             - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})

    # Apply non-local geometric tiling reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Generate asymmetric spatial tiling with radial influence
        tiling_hash = np.random.rand(n, 2) * 0.1
        for i in range(n):
            v[3*i] += tiling_hash[i, 0] * (radii[i] / np.mean(radii))
            v[3*i+1] += tiling_hash[i, 1] * (radii[i] / np.mean(radii))

        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Apply targeted reconfiguration on least constrained node with dynamic expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Compute distance matrix using numpy broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum distance to others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Targeted radius expansion with soft constraints and asymmetric growth
        current_total = np.sum(radii)
        growth_factor = 0.006  # Conservative growth target
        # Compute expansion factor based on current distribution of space
        total_space = np.sum(np.min(dists, axis=1))
        expansion_factor = (growth_factor * (current_total / total_space)) / (n - 1)
        
        new_radii = radii.copy()
        # Exponentially expand the least constrained circle
        new_radii[least_constrained_idx] = radii[least_constrained_idx] * 1.1 + expansion_factor * 1.6
        # Distribute the remaining expansion proportionally to avoid excessive growth
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1 + 0.2 * np.random.rand())

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
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Final optimization with tighter tolerances and improved convergence
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Compute distance matrix for validation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Apply additional spatial perturbation to enhance configuration diversity
        perturbation = np.random.rand(n, 2) * 0.01 * (radii / np.max(radii))
        for i in range(n):
            v[3*i] += perturbation[i, 0]
            v[3*i+1] += perturbation[i, 1]

        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())