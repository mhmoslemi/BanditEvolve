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
        x = x_center + np.random.uniform(-0.08, 0.08)
        y = y_center + np.random.uniform(-0.08, 0.08)
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
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-11, "eps": 1e-11})
    
    # Asymmetric reconfiguration: apply stochastic spatial perturbation
    if res.success:
        v = res.x
        # Apply directional spatial perturbation for better exploration
        spatial_hash = np.random.rand(n, 2) * 0.1
        perturbed_v = v.copy()
        for i in range(n):
            # Introduce more significant horizontal perturbation for row-based clustering
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]

        # Re-evaluate with perturbed spatial parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-11})
    
    # Targeted radius expansion on least constrained circle with advanced selection criteria
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances using vectorization
        dx, dy = np.meshgrid(centers[:, 0], centers[:, 0])
        dx -= centers[:, 0, np.newaxis]
        dy -= centers[:, 1, np.newaxis]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find isolation metric as sum of distances to neighbors, but avoid self
        # Use masked array to skip diagonal entries
        mask = np.eye(n, dtype=bool)
        dists_masked = np.ma.masked_array(dists, mask=mask)
        isolation = np.sum(dists_masked, axis=1)
        least_constrained_idx = np.argmax(isolation)
        
        # Calculate expansion with spatial constraint preservation
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.008  # Increase expansion goal
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Use a more refined expansion strategy: base on spatial availability
        new_radii = radii.copy()
        
        # Apply radial expansion with directional guidance towards isolated circles
        # First expand the least constrained circle
        for i in range(n):
            # Use directional expansion, prioritizing isolated ones
            if i == least_constrained_idx:
                # Double expansion for isolated circle to maximize gain
                new_radii[i] = np.clip(radii[i] + expansion_factor * 1.5, 1e-4, 0.5)
            else:
                # Moderate expansion for others
                new_radii[i] = np.clip(radii[i] + expansion_factor * 0.8, 1e-4, 0.5)
        
        # Apply expansion and re-evaluate
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-11})
    
    # Final configuration check and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())