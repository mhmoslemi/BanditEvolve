import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Fix random seed for reproducibility
    np.random.seed(123456)
    
    # Initialize positions using a randomized grid pattern
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add small randomness to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.03, 0.03)
        y = y_center + np.random.uniform(-0.03, 0.03)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds for all circles: x,y in [0,1], radius in [1e-4, 0.5]
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-12})

    # Asymmetric spatial reconfiguration with geometric hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate geometric hash for controlled spatial perturbation
        spatial_hash = np.random.rand(n, 2) * 0.03 - 0.015  # Small perturbation
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += spatial_hash[i, 0]
            new_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 800, "ftol": 1e-11, "eps": 1e-12})

    # Targeted radius expansion on least constrained circle with controlled expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle: one with maximum minimum distance to other circles
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)  # Most underutilized
        
        # Calculate expansion with total-sum constraint and spatial constraints
        target_total_sum = np.sum(radii) + 0.0075  # 0.75% increase
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
        
        # Apply expansion with soft spatial enforcement
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Controlled over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.1 * np.random.rand())  # Stochastic expansion
        
        # Apply expansion with constraint validation
        valid = False
        exp_v = v.copy()
        exp_v[2::3] = new_radii
        exp_centers = np.column_stack([exp_v[0::3], exp_v[1::3]])
        
        # Validate expanded configuration
        for i in range(n):
            for j in range(i + 1, n):
                dx = exp_centers[i, 0] - exp_centers[j, 0]
                dy = exp_centers[i, 1] - exp_centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < new_radii[i] + new_radii[j] - 1e-12:
                    new_radii = radii + (new_radii - radii) * 0.9  # Reduce expansion slightly
                    valid = False
                    break
            if not valid:
                break
        valid = True  # If no overlap found
            
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())