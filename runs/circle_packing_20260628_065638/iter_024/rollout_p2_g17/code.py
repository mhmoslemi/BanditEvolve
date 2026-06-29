import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with asymmetric spatial hashing and grid disruption
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols + np.random.uniform(-0.1, 0.1)
        y_center = (row + 0.5) / rows + np.random.uniform(-0.1, 0.1)
        # Introduce asymmetry and clustering by row
        if row % 2 == 1:
            x_center += np.random.uniform(-0.15, 0.15)
            y_center += np.random.uniform(-0.2, 0.2)
        xs.append(x_center)
        ys.append(y_center)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n matches v0

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
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
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})
    
    # Asymmetric spatial reconfiguration with dynamic hash grid
    if res.success:
        v = res.x
        # Apply randomized dynamic hashing
        hash_grid = np.random.rand(n, 2) * 0.1
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_grid[i, 0]
            perturbed_v[3*i+1] += hash_grid[i, 1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted radius expansion on two under-constrained circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized pairwise distance calculation
        dists = np.sqrt(((centers[:, np.newaxis] - centers[np.newaxis, :]) ** 2).sum(axis=2))
        
        # Calculate min pairwise distances and find under-constrained circles
        min_dists = np.min(dists, axis=1)
        under_constrained_idx = np.argsort(min_dists)[:2]  # Select top 2 least constrained
        
        # Calculate expansion factor to increase radii
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.01
        expansion_factor = (target_total_sum - total_sum) / n  # Distribute expansion more evenly
        
        # Apply controlled expansion to both under-constrained circles
        new_radii = radii.copy()
        new_radii[under_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            new_radii[i] += expansion_factor  # Apply to all
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())