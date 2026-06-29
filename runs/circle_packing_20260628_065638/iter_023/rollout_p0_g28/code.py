import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized grid and staggered staggering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset with decreasing variance to avoid symmetry
        x = x_center + np.random.uniform(-0.05, 0.05) * (1 - 0.8 * (row + col) / (rows + cols))
        y = y_center + np.random.uniform(-0.05, 0.05) * (1 - 0.8 * (row + col) / (rows + cols))
        # Staggered grid by alternating row shifts
        if row % 2 == 1:
            x += 0.5 / cols * (1 - 0.3 * (row + col) / (rows + cols))
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
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing and adaptive penalty
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                # Compute distance squared between centers
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                distance_sq = dx * dx + dy * dy
                # Compute sum of radii squared with soft penalty 
                radii_sum = v[3*i+2] + v[3*j+2]
                return distance_sq - radii_sum * radii_sum + 1e-8 * (distance_sq - radii_sum * radii_sum)
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "gtol": 1e-10})
    
    # Apply hybrid reconfiguration: geometric hashing with radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create randomized geometric hashing pattern with adaptive perturbation
        random_hash = np.random.rand(n, 2) * 0.03
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0] * (1 - 0.8 * (i / n))
            perturbed_v[3*i+1] += random_hash[i, 1] * (1 - 0.8 * (i / n))
        
        # Re-evaluate with new perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-10})
    
    # Targeted radius expansion with constrained total sum for novel configurations
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Identify circle with smallest non-zero radius
        smallest_radius_idx = np.argmin(radii[radii > 1e-6])
        # Calculate expansion factor to increase total sum while keeping it below threshold
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.008
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Create new radii vector with adaptive expansion
        new_radii = radii.copy()
        new_radii[smallest_radius_idx] += expansion_factor * 1.2  # Over-expansion to trigger reconfiguration
        for i in range(n):
            if i != smallest_radius_idx:
                new_radii[i] += expansion_factor * (1 - (i / n))  # Adaptive expansion decay
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())