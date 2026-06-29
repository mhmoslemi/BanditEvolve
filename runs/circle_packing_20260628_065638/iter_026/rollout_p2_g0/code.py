import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with a structured grid with randomized offset and row staggering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Small random offset for diversification
        x = x_center + np.random.uniform(-0.06, 0.06)
        y = y_center + np.random.uniform(-0.06, 0.06)
        # Shift alternate rows for staggered layout
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radius based on spacing and padding
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n constraints

    # Objective function
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Define boundary constraints using anonymous functions with capture
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Build all pairwise non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Use lambda with fixed parameters to avoid closure issues
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-12, "eps": 1e-10})

    # Asymmetric reconfiguration: apply full spatial perturbation
    if res.success:
        v = res.x
        # Random spatial perturbation from uniform distribution
        spatial_hash = np.random.rand(n, 2) * 0.2  # Increased range for bigger changes
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-run optimization with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-10})

    # Targeted radius expansion on least constrained circle using vectorized computation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])

        # Vectorized distance matrix for all pairs
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Find the least constrained circle (minimum min distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Calculate expansion amount with target growth
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.006  # Conservatively increase
        expansion_factor = (target_total_sum - total_sum) / (n - 1)

        # Apply expansion with soft limits to avoid invalid overlaps
        new_radii = radii.copy()
        for i in range(n):
            if i == least_constrained_idx:
                new_radii[i] = np.clip(radii[i] + expansion_factor * 1.15, 1e-4, 0.5)
            else:
                new_radii[i] = np.clip(radii[i] + expansion_factor * 0.8, 1e-4, 0.5)

        # Re-run with new radii and validate constraints
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-10})

    # Final fallback to initial state if optimization failed
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())