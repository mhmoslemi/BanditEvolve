import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with dense grid with added randomized offsets and staggered rows
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Introduce spatial perturbation to break symmetry
        x = x_center + np.random.uniform(-0.01, 0.01)
        y = y_center + np.random.uniform(-0.01, 0.01)
        
        # Stagger alternate rows to avoid regular grid patterns
        if row % 2 == 1:
            x += 0.25 / cols
        
        xs.append(x)
        ys.append(y)
    
    # Initialize radii based on compact packing estimates
    r0 = 0.25 / cols - 1e-3
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

    # Initial optimization with increased max iterations and tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-10})
    
    # Apply hybrid reconfiguration: geometric hashing for spatial exploration and constrained radius expansion
    if res.success:
        v = res.x
        
        # Create geometric hash for perturbation
        hash_vec = np.random.rand(n, 2) * 0.1
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_vec[i, 0]
            perturbed_v[3*i+1] += hash_vec[i, 1]
        
        # Re-optimization with perturbed coordinates and same constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
    
    # Targeted radius expansion on the circle with smallest non-zero radius, while maintaining sum constraint
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances to identify least constrained circle
        dists = np.zeros((n, n))
        for i in range(n):
            dx = centers[i, 0] - centers[:, 0]
            dy = centers[i, 1] - centers[:, 1]
            dists[i] = np.sqrt(dx**2 + dy**2)
        
        # Identify the circle with smallest radius and largest minimum distance to neighbors
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argsort(np.column_stack((radii, min_dists)))[-1][0]
        
        # Calculate expansion factor while maintaining total sum constraint
        current_sum = np.sum(radii)
        target_sum = current_sum + 0.008  # Increase by 0.008
        expansion_factor = (target_sum - current_sum) / (n - 1)
        
        # Apply expansion to the least constrained circle and all others
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Update decision vector and re-evaluate with new configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-optimization with expanded radii and same constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())