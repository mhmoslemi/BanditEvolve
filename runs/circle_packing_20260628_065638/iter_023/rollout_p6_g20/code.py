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
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.4 / cols - 1e-3
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

    # Initial optimization with more aggressive perturbations and tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1600, "ftol": 1e-11})
    
    # Asymmetric reconfiguration with geometric hashing and radius expansion
    if res.success:
        v = res.x
        # Compute initial centers and radii
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Generate geometric hash perturbation with higher variance
        geometric_hash = np.random.rand(n, 2) * 0.15
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += geometric_hash[i, 0]
            perturbed_v[3*i+1] += geometric_hash[i, 1]
        
        # Re-evaluate with perturbed configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
    
    # Trigger asymmetric reconfiguration with geometric hashing and controlled radius expansion
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute pairwise dists for adjacency analysis
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Find the circle with the minimal non-zero radius
        min_radii_idx = np.argmin(radii)
        min_radius = radii[min_radii_idx]
        
        # Find the circle with the largest non-overlapping spacing
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Generate a geometric hash for asymmetric reconfiguration
        reconfig_hash = np.random.rand(n, 2) * 0.2
        v_new = v.copy()
        for i in range(n):
            v_new[3*i] += reconfig_hash[i, 0]
            v_new[3*i+1] += reconfig_hash[i, 1]
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Targeted radius expansion with gradient-aided refinement
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute pairwise distance array
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Find least constrained circle using combined radii and spacing
        min_radii_idx = np.argmin(radii)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Compute gradient of sum_radii with respect to least constrained circle
        # Using finite-difference approximation
        eps = 1e-8
        gradient = np.zeros(n)
        for i in range(n):
            perturb = np.zeros(3*n)
            perturb[3*i+2] = eps
            perturbed_v = v.copy()
            perturbed_v[3*i+2] += eps
            perturbed_res = minimize(neg_sum_radii, perturbed_v, method="SLSQP",
                                     bounds=bounds, constraints=cons, options={"maxiter": 1, "ftol": 1e-10})
            if perturbed_res.success:
                grad = (neg_sum_radii(perturbed_v) - neg_sum_radii(v)) / eps
                gradient[i] = grad
        
        # Find direction of maximum radius increase
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1e-6:
            direction = gradient / grad_norm
        else:
            direction = np.zeros(n)
        
        # Apply controlled expansion to all circles with direction-based weighting
        expansion_factor = 0.012
        expanded_radii = radii.copy()
        expanded_radii += direction * expansion_factor
        
        # Ensure minimal radius remains above threshold
        expanded_radii = np.maximum(expanded_radii, 1e-4)
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = expanded_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())