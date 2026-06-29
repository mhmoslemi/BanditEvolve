import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize with a structured grid with randomized offset and staggered layout
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        x_rand = np.random.uniform(-0.1, 0.1)
        y_rand = np.random.uniform(-0.1, 0.1)
        # Stagger alternate rows for better spacing
        if row % 2 == 1:
            x_center += 0.5 / cols
        x = x_center + x_rand
        y = y_center + y_rand
        xs.append(x)
        ys.append(y)
    
    # Initial radius estimate: based on grid spacing and padding
    r0 = 0.39 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds consistent with 3*n parameters
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * n

    # Objective to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Boundary constraints: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    cons = []
    for i in range(n):
        # X-bound constraints
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Y-bound constraints
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints using vectorized distance calculation
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with capture to avoid closure issues
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization with high precision and multiple passes
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "eps": 1e-10})
    
    # Asymmetric reconfiguration with spatial stochasticity
    if res.success:
        v = res.x
        # Create stochastic spatial perturbation based on geometric hash
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-10})

    # Targeted radius expansion on least constrained circle with soft reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix using broadcasting for better performance
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate isolation metric: minimum distance to other centers
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate safe expansion while respecting boundaries
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.008  # Targeted expansion
        # Use gradient-based scaling for radius expansion
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Conservative expansion to avoid collision
        new_radii = radii.copy()
        expanded_v = v.copy()
        
        # Expand isolation circle first with higher priority
        new_radii[least_constrained_idx] = np.clip(radii[least_constrained_idx] + expansion_factor * 1.2, 1e-4, 0.5)
        
        # Apply moderate expansion to others with safety margin
        for i in range(n):
            if i != least_constrained_idx:
                # Introduce small stochastic variance for spatial diversity
                expand_amount = expansion_factor * (1 + 0.1 * np.random.rand())
                new_radii[i] = np.clip(radii[i] + expand_amount, 1e-4, 0.5)
        
        # Apply expansion and re-optimize
        expanded_v[2::3] = new_radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "eps": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())