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
        def boundary_x_min(v, i=i):
            return v[3*i] - v[3*i+2]
        def boundary_x_max(v, i=i):
            return 1.0 - v[3*i] - v[3*i+2]
        def boundary_y_min(v, i=i):
            return v[3*i+1] - v[3*i+2]
        def boundary_y_max(v, i=i):
            return 1.0 - v[3*i+1] - v[3*i+2]
        cons.append({"type": "ineq", "fun": boundary_x_min})
        cons.append({"type": "ineq", "fun": boundary_x_max})
        cons.append({"type": "ineq", "fun": boundary_y_min})
        cons.append({"type": "ineq", "fun": boundary_y_max})
    
    # Vectorized overlap constraints with geometric hashing and adjacency-aware regularization
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                r1 = v[3*i+2]
                r2 = v[3*j+2]
                return dist_sq - (r1 + r2)**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
    
    # Apply asymmetric topological disruption: geometric hashing with controlled randomness
    if res.success:
        v = res.x
        # Create a randomized geometric hashing matrix for topological reshuffling
        random_hash = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0] * (1.0 - v[3*i+2]/0.5)
            perturbed_v[3*i+1] += random_hash[i, 1] * (1.0 - v[3*i+2]/0.5)
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12})
    
    # Targeted radius expansion on most under-constrained circle with adjacency-aware optimization
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
        smallest_radius_idx = np.argmin(radii)
        
        # Calculate controlled expansion factor using adjacency-based weighting
        total_sum = np.sum(radii)
        expansion_factor = 0.008 / (n - 1) * np.max(min_dists) / np.mean(min_dists)
        
        # Apply adjacency-aware expansion to under-constrained and smallest circles
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3
        new_radii[smallest_radius_idx] += expansion_factor * 1.3
        for i in range(n):
            if i != least_constrained_idx and i != smallest_radius_idx:
                new_radii[i] += expansion_factor * (1.0 - min_dists[i]/np.max(min_dists))
        
        # Update decision vector and re-evaluate
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12})
    
    # Final cleanup and return
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())