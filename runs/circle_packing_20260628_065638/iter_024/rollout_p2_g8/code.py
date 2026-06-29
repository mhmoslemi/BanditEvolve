import numpy as np

def run_packing():
    n = 26
    
    # Initialize with geometric hashing and staggered grid
    cols = 5
    rows = (n + cols - 1) // cols
    
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
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
    
    # Vectorized overlap constraints using geometric hashing
    for i in range(n):
        for j in range(i+1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    if res.success:
        v = res.x
        
        # Apply randomized geometric hashing for asymmetric topological disruption
        hash_grid = np.random.rand(n, 2) * 0.06
        v_perturbed = v.copy()
        for i in range(n):
            v_perturbed[3*i] += hash_grid[i, 0]
            v_perturbed[3*i+1] += hash_grid[i, 1]
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute pairwise distances and select the most under-constrained circle
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Average minimum distance
        min_dists = np.min(dists, axis=1)
        average_min_dist = np.mean(min_dists)
        under_constrained_indices = np.where(min_dists < average_min_dist * 1.2)[0]
        most_under_constrained_idx = under_constrained_indices[0]
        
        # Calculate controlled expansion to increase radius of most under-constrained circle
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.008
        expansion_factor = (target_total_sum - total_sum) / n
        
        # Apply controlled expansion to most under-constrained circle
        new_radii = radii.copy()
        new_radii[most_under_constrained_idx] += expansion_factor * 1.3
        for i in range(n):
            if i != most_under_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector and re-evaluate
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())