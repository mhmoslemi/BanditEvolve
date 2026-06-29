import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Adaptive initialization with geometric clustering, staggered grid, and randomized perturbations
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset with adaptive range based on row
        x_offset = np.random.uniform(-0.03 + 0.002 * row, 0.03 + 0.002 * row)
        y_offset = np.random.uniform(-0.03 + 0.001 * row, 0.03 + 0.001 * row)
        x = x_center + x_offset
        y = y_center + y_offset
        # Staggered row shift for hexagonal packing
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.38 / cols - 1e-3
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
    
    # Vectorized overlap constraints with geometric hashing using matrix operations
    # Precompute all pairwise index combinations using broadcasting
    i_indices, j_indices = np.triu_indices(n, 1)
    i_indices = i_indices.astype(int)
    j_indices = j_indices.astype(int)
    
    # Create a matrix of all pairwise distance squared - (r_i + r_j)^2
    def constraint_func(v):
        dx = v[3*i_indices] - v[3*j_indices]
        dy = v[3*i_indices+1] - v[3*j_indices+1]
        ri = v[3*i_indices+2]
        rj = v[3*j_indices+2]
        return dx*dx + dy*dy - (ri + rj)**2
    
    # Create constraints for all pairs
    for i, j in zip(i_indices, j_indices):
        cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func(v)[np.where((i_indices == i) & (j_indices == j))[0][0]]})

    # Initial optimization with tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Adaptive geometric perturbation for critical circle reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Isolate the three most constrained circles (least spatial freedom)
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        min_dists = np.min(dists, axis=1)
        most_constrained_indices = np.argsort(min_dists)[:3]
        
        # Isolate and reconfigure these circles with spatial constraints
        # Create a perturbation matrix for these circles
        perturbation = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in most_constrained_indices:
            perturbed_v[3*i] += perturbation[i, 0]
            perturbed_v[3*i+1] += perturbation[i, 1]
        
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion on least constrained circle with geometric awareness
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor with geometric awareness
        total_sum = np.sum(radii)
        # Expand the least constrained circle with 0.008 increment
        target_total_sum = total_sum + 0.008
        expansion_factor = (target_total_sum - total_sum) / (n)
        
        # Distribute expansion while preserving spatial constraints
        # Prioritize expansion for circles with largest minimum distance
        expanded_radii = radii.copy()
        expanded_radii[least_constrained_idx] += expansion_factor * 1.2  # Prioritize least constrained
        for i in range(n):
            expanded_radii[i] += expansion_factor * (min_dists[i] / np.max(min_dists))  # Spatially aware expansion
        
        # Update decision vector with expanded radii
        v_expanded = v.copy()
        v_expanded[2::3] = expanded_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Final optimization pass with geometric awareness and controlled expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Final radius expansion of least constrained circle with geometric awareness
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.008
        expansion_factor = (target_total_sum - total_sum) / (n)
        
        # Distribute expansion while preserving spatial constraints
        expanded_radii = radii.copy()
        expanded_radii[least_constrained_idx] += expansion_factor * 1.2  # Prioritize least constrained
        for i in range(n):
            expanded_radii[i] += expansion_factor * (min_dists[i] / np.max(min_dists))  # Spatially aware expansion
        
        # Update decision vector with expanded radii
        v_expanded = v.copy()
        v_expanded[2::3] = expanded_radii
        
        # Re-evaluate with final expansion and configuration
        res = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())