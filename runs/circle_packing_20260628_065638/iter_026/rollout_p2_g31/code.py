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
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.1, 0.1)
        y = y_center + np.random.uniform(-0.1, 0.1)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.25 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.32 / cols
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.45)]

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

    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})

    # Asymmetric reconfiguration step with randomized geometric hashing
    if res.success:
        v = res.x
        # Stochastic geospatial hashing for reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        # Re-evaluate with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion with isolation-based prioritization
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Efficient vectorized distance matrix calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx ** 2 + dy ** 2)
        
        # Find the circle with the largest area of isolation
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion amount with soft enforcement and adaptive scaling
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.0085  # Incremental gain target
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Apply expansion with adaptive scaling and spatial constraints
        new_radii = radii.copy()
        expanded_v = v.copy()
        
        # Exploit isolation for targeted expansion
        expansion_amount = expansion_factor * np.random.uniform(1.2, 1.4)
        new_radii[least_constrained_idx] = np.clip(radii[least_constrained_idx] + expansion_amount, 1e-4, 0.45)
        
        # Apply controlled expansion to other circles
        for i in range(n):
            if i != least_constrained_idx:
                expansion_amount = expansion_factor * np.random.uniform(0.6, 1.2)
                new_radii[i] = np.clip(radii[i] + expansion_amount, 1e-4, 0.45)
        
        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_ = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_ = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_**2 + dy_**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Gradually reduce expansion if invalid
                expansion_decay = np.random.uniform(0.8, 0.95)
                new_radii = radii + (new_radii - radii) * expansion_decay
        
        # Update and re-evaluate
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    # Apply final spatial stabilization to avoid oscillation
    if res.success:
        v = res.x
        spatial_hash = np.random.rand(n, 2) * 0.04
        stabilized_v = v.copy()
        for i in range(n):
            stabilized_v[3*i] += spatial_hash[i, 0]
            stabilized_v[3*i+1] += spatial_hash[i, 1]
        
        res = minimize(neg_sum_radii, stabilized_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 100, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())