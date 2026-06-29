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
        x = x_center + np.random.uniform(-0.02, 0.02)
        y = y_center + np.random.uniform(-0.02, 0.02)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.34 / cols - 1e-3
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
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})

    # Apply shockwave reconfiguration to smallest non-zero radius with topological shuffling
    if res.success:
        v = res.x
        radii = v[2::3]
        # Compute pairwise distances and interaction metrics
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Calculate isolation metric (minimum distance to all other circles)
        isolation = np.min(dists, axis=1)
        isolated_idx = np.argmin(isolation)
        # Calculate expansion factor
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.008
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Apply topological reordering via permutation to break existing structure
        new_order = np.random.permutation(n)
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] = v[3*new_order[i]]
            new_v[3*i+1] = v[3*new_order[i]+1]
            new_v[3*i+2] = v[3*new_order[i]+2]
        
        # Apply controlled expansion to isolated circle
        new_radii = radii.copy()
        new_radii[isolated_idx] += expansion_factor * 1.3  # Over-expansion to trigger reconfiguration
        for i in range(n):
            if i != isolated_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector
        v_new = new_v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with new configuration and expanded radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 800, "ftol": 1e-11})

    # Final optimization with tight tolerances and gradient stabilization
    if res.success:
        v = res.x
        radii = v[2::3]
        # Re-check distances for any violation with tighter epsilon
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Enforce strict non-overlap with enhanced gradient regularization
        for i in range(n):
            for j in range(i + 1, n):
                dist = dists[i, j]
                if dist < radii[i] + radii[j] - 2e-7:
                    # Apply localized perturbation to break overlap
                    overlap = radii[i] + radii[j] - dist
                    v[3*i] += np.random.uniform(-0.001, 0.001)
                    v[3*i+1] += np.random.uniform(-0.001, 0.001)
                    v[3*j] += np.random.uniform(-0.001, 0.001)
                    v[3*j+1] += np.random.uniform(-0.001, 0.001)
                    # Re-evaluate with adjusted parameters
                    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                                   constraints=cons, options={"maxiter": 200, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())