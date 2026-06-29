import numpy as np

def run_packing():
    n = 26
    # Asymmetric topological disruption: randomized geometric hashing
    cols = int(np.ceil(np.sqrt(n)))
    xs = np.zeros(n)
    ys = np.zeros(n)
    for i in range(n):
        col = i % cols
        row = i // cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / (cols + 1)  # Introduce asymmetric row scaling
        xs[i] = x_center + np.random.uniform(-0.05, 0.05)
        ys[i] = y_center + np.random.uniform(-0.05, 0.05)
        if row % 2 == 1:
            xs[i] += 0.5 / cols  # Staggered grid for asymmetry
        # Introduce asymmetric randomness for circle placement
        if i < n//2:
            xs[i] += np.random.uniform(-0.02, 0.02)
            ys[i] += np.random.uniform(-0.02, 0.02)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n bounds

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    for i in range(n):
        # Boundary constraints with vectorized lambda
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Asymmetric overlap constraints with geometric hashing and stochastic constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_squared = dx*dx + dy*dy
                # Use asymmetric penalty for overlapping circles
                if i < j:
                    radius_sum = v[3*i+2] + v[3*j+2] * 1.1  # Asymmetric constraint
                else:
                    radius_sum = v[3*i+2] * 1.1 + v[3*j+2]
                return dist_squared - radius_sum*radius_sum
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with higher tolerance for asymmetric constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 3000, "ftol": 1e-11, "gtol": 1e-9})
    
    # Introduce asymmetric reconfiguration: target the least constrained circle with topological shift
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances and find adjacency relationships with asymmetric weighting
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Calculate adjacency matrix with asymmetric distance thresholds
        adjacency_matrix = dists < 1.5 * (np.sqrt(np.sum((centers[i] - centers[j])**2)) - radii[i] - radii[j] - 1e-6)
        
        # Find the most under-constrained circle using asymmetric metric
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)  # Asymmetric selection of least constrained
        
        # Calculate controlled expansion factor with asymmetric scaling
        total_sum = np.sum(radii)
        expansion_factor = 0.01 / (n - 1)  # Asymmetric expansion for topological reconfiguration
        
        # Apply controlled expansion to the most under-constrained circle and adjacent ones
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 2  # Asymmetric expansion
        for i in range(n):
            if i != least_constrained_idx and adjacency_matrix[least_constrained_idx, i]:
                new_radii[i] += expansion_factor
        
        # Update decision vector with asymmetric adjustment
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-9})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())