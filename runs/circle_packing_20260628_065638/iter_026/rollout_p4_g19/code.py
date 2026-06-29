import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with randomized geometric hashing and advanced staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Random offset with exponential fading for better distribution
        x_offset = np.random.rand() * 0.08
        x = x_center + x_offset * (1.0 - np.exp(-np.random.rand() * 2))
        y_offset = np.random.rand() * 0.08
        y = y_center + y_offset * (1.0 - np.exp(-np.random.rand() * 2))
        # Shift alternate rows with adaptive amount
        if row % 2 == 1:
            x += 0.5 / cols * (1.0 - np.random.rand() * 0.5)
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds with strict constraint consistency
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints with optimized lambda closures
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with efficient broadcasting
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tighter tolerances and higher precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-12, "maxls": 200})

    # Execute geometric hashing transformation for constrained reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create geometric hash for spatial reconstruction
        geom_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += geom_hash[i, 0] * (1.0 - np.exp(-0.2 * np.random.rand()))
            perturbed_v[3*i+1] += geom_hash[i, 1] * (1.0 - np.exp(-0.2 * np.random.rand()))
        
        # Re-evaluate with geometric hash perturbation
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "maxls": 200})
    
    # Execute targeted radius expansion on smallest isolated circle with constraint enforcement
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix computation with broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Compute isolation metric as average reciprocal distance to others
        isolation_metric = 1.0 / (1e-6 + np.mean(1.0 / (dists + 1e-6), axis=1))
        isolated_idx = np.argmin(isolation_metric)  # Smallest isolation implies largest isolation

        # Calculate expansion factor with controlled spatial expansion
        target_total_sum = np.sum(radii) + 0.006
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)

        # Apply expansion with spatial constraint preservation
        new_radii = radii.copy()
        new_radii[isolated_idx] += expansion_factor * (1.2 + np.random.rand() * 0.2)
        for i in range(n):
            if i != isolated_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.15 * np.random.rand())

        # Create perturbed decision vector and re-evaluate under constraints
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "maxls": 200})
    
    # Final optimization with adaptive constraint enforcement
    if res.success:
        v_final = res.x
        radii_final = v_final[2::3]
        centers_final = np.column_stack([v_final[0::3], v_final[1::3]])
        
        # Add soft spatial constraints for robustness
        soft_cons = []
        for i in range(n):
            for j in range(i + 1, n):
                def soft_constraint_func(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2 + 1e-8
                soft_cons.append({"type": "ineq", "fun": soft_constraint_func})
        
        res = minimize(neg_sum_radii, v_final, method="SLSQP", bounds=bounds,
                       constraints=cons + soft_cons, options={"maxiter": 400, "ftol": 1e-12, "maxls": 200})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())