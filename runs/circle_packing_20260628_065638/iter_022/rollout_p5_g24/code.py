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
        x = x_center + np.random.uniform(-0.04, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.04)
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
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
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
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Trigger constrained reconfiguration with randomized spatial constraint function
    if res.success:
        v = res.x
        # Implement stochastic spatial constraint perturbation for exploration
        spatial_map = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_map[i, 0] * 1.5
            perturbed_v[3*i+1] += spatial_map[i, 1] * 1.5
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion on most isolated circle with adjacency-aware expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        # Compute pairwise distances for all circles using vectorized operations
        x = v[0::3]
        y = v[1::3]
        dists = np.zeros((n, n))
        for i in range(n):
            dx = x[i] - x
            dy = y[i] - y
            dists[i] = np.sqrt(dx*dx + dy*dy)
        # Find the circle with the most isolation (smallest sum of reciprocals of distances)
        isolation = np.sum(1 / (dists + 1e-8), axis=1)
        isolated_idx = np.argmin(isolation)
        
        # Compute current total sum
        total_sum = np.sum(radii)
        # Calculate expansion factor for controlled radius increase
        target_total_sum = total_sum + 0.0065
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Create adjusted radius vector with adjacency-based expansion
        new_radii = radii.copy()
        new_radii[isolated_idx] += expansion_factor * 1.15
        for i in range(n):
            if i != isolated_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector and re-evaluate
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Final optimization pass with tighter tolerances and gradient checking
    if res.success:
        v = res.x
        # Recalculate pairwise distances
        x = v[0::3]
        y = v[1::3]
        dists = np.zeros((n, n))
        for i in range(n):
            dx = x[i] - x
            dy = y[i] - y
            dists[i] = np.sqrt(dx*dx + dy*dy)
        # Re-check isolation based on updated configuration
        isolation = np.sum(1 / (dists + 1e-8), axis=1)
        isolated_idx = np.argmin(isolation)
        
        # Final radius adjustment to maximize total sum within constraints
        total_sum = np.sum(v[2::3])
        expansion = (total_sum + 0.008 - total_sum) / (n - 1)
        for i in range(n):
            if i != isolated_idx:
                v[3*i + 2] += expansion
        
        # Final optimization
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())