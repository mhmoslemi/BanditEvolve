import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize positions with randomized geometric clustering, staggered grid, and adaptive spacing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset for diversity in initial configuration
        x = x_center + np.random.uniform(-0.04, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.04)
        # Staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        # Ensure small spacing to allow expansion later
        if col == 0 or col == cols-1:
            x += np.random.uniform(-0.02, 0.02)
        if row == 0 or row == rows-1:
            y += np.random.uniform(-0.02, 0.02)
        xs.append(x)
        ys.append(y)
    
    # Initial radius assignment with adaptive scaling
    r0 = 0.35 / cols - 1e-3
    r0 = np.clip(r0, 1e-4, 0.4)  # Clamp to ensure non-zero radii
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Must match 3*n length

    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective to maximize sum of radii

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints using matrix operations for efficiency
    i_indices, j_indices = np.triu_indices(n, 1)
    i_indices = i_indices.astype(int)
    j_indices = j_indices.astype(int)
    
    def constraint_func(v):
        dx = v[3*i_indices] - v[3*j_indices]
        dy = v[3*i_indices+1] - v[3*j_indices+1]
        ri = v[3*i_indices+2]
        rj = v[3*j_indices+2]
        return dx*dx + dy*dy - (ri + rj)**2
    
    # Create constraints for all pairs
    for i, j in zip(i_indices, j_indices):
        cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func(v)[np.where((i_indices == i) & (j_indices == j))[0][0]]})

    # First optimization with moderate settings
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})

    # Induce geometric reconfiguration via randomized perturbation
    if res.success:
        v = res.x
        # Generate a geometric hash map for controlled spatial reordering
        random_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        # Re-evaluate with perturbed parameters
        res_perturb = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
        res = res_perturb if res_perturb.success else res

    # Adaptive expansion of least constrained circle with controlled radii update
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Find the circle with smallest non-zero radius
        min_radius_idx = np.argmin(radii)

        # Calculate pairwise distances for constraint analysis
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)

        # Find least constrained circle (largest minimum distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Calculate current total sum
        total_sum = np.sum(radii)
        # Calculate expansion factor for controlled radius increase
        target_total_sum = total_sum + 0.008
        expansion_factor = (target_total_sum - total_sum) / (n - 1)

        # Create adjusted radius vector with adjacency-based expansion
        new_radii = radii.copy()
        new_radii[min_radius_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != min_radius_idx:
                new_radii[i] += expansion_factor

        # Update decision vector and re-evaluate with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Final optimization pass with tighter tolerances and stricter constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.006
        expansion = (target_total_sum - total_sum) / (n - 1)
        for i in range(n):
            if i != least_constrained_idx:
                v[3*i+2] += expansion
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())