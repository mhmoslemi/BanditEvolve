import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols

    # Initialize positions using a staggered grid with randomized geometric hashing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset + geometric hashing for spatial diversity
        x_offset = np.random.uniform(-0.06, 0.06) * (1.0 / cols ** 0.5)
        y_offset = np.random.uniform(-0.06, 0.06) * (1.0 / rows ** 0.5)
        x = x_center + x_offset
        y = y_center + y_offset
        # Stagger alternate rows for better spacing
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with a larger base and dynamic factor that depends on spacing
    r0 = 0.4 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define consistent length bounds
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective function to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Boundary constraints
    cons = []
    for i in range(n):
        # Left side with radius constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right side with radius constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom side with radius constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top side with radius constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Overlap constraints with vectorized broadcasting
    # Precompute distance matrix and handle non-overflowed calculations
    def constraint_func(v, i, j):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func(v, i, j)})

    # Initial optimization with tight tolerances and high iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "eps": 1e-10})

    # Asymmetric reconfiguration with spatial perturbation matrix
    if res.success:
        v = res.x
        # Generate a spatial perturbation matrix using random geometric hashing
        perturbation_matrix = np.random.rand(n, 2) * 0.035
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += perturbation_matrix[i, 0]
            perturbed_v[3*i+1] += perturbation_matrix[i, 1]
        
        # Re-evaluate with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-10})

    # Targeted radius expansion on most-isolated circle with spatial gradient analysis
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Compute spatial gradient matrix without overflow
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx*dx + dy*dy)
        
        # Identify least constrained circle using spatial isolation metric
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Apply targeted expansion with adaptive constraint handling
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.0088  # Incremental improvement target
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Soft expansion with adaptive adjustment
        new_radii = radii.copy()
        for i in range(n):
            if i == least_constrained_idx:
                # Slight over-expansion to potentially unlock new configuration
                new_radii[i] += expansion_factor * 1.2
            else:
                # Moderate expansion to improve overall density
                new_radii[i] += expansion_factor * 0.9

        # Apply expansion and ensure constraints are still valid
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii

        # Iterative constraint checking with soft backtracking to ensure feasibility
        while True:
            res_expanded = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                                    constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-10})
            if res_expanded.success:
                v = res_expanded.x
                break
            else:
                # Backtrack by reducing expansion if constraints fail
                new_radii = radii + (new_radii - radii) * 0.99
                expanded_v[2::3] = new_radii
        
        # Final post-expansion refinement
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-10})

    # Final output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())