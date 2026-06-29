import numpy as np

def run_packing():
    n = 26
    cols = 5  # Optimal for staggered grid with 5 columns
    rows = (n + cols - 1) // cols  # Calculate rows for staggered grid
    
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
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Length 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})  # x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})  # 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})  # y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})  # 1 - y - r >= 0
    
    # Vectorized overlap constraints with geometric hashing
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                radii_sum = v[3*i+2] + v[3*j+2]
                return dist_sq - radii_sum**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tight tolerances and more iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-9})
    
    # Asymmetric topological disruption: geometric hashing with adjacency constraints
    if res.success:
        v = res.x
        # Generate a randomized geometric hash map to disrupt the current topology
        hash_map = np.random.rand(n, 2) * 0.08  # Small displacement factor
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_map[i, 0]
            perturbed_v[3*i+1] += hash_map[i, 1]
        
        # Add adjacency constraints with minimal distance penalty to force reordering
        for i in range(n):
            for j in range(i + 1, n):
                # Enforce minimal distance between adjacent circles with slight penalty
                cons.append({"type": "ineq",
                             "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2 + 1e-7})
        
        # Re-evaluate with perturbed parameters and adjacency constraints
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-9})
    
    # Targeted radius expansion on the most under-constrained circle
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Precompute pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Find the most under-constrained circle (least constrained by other circles)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # Calculate controlled expansion factor to unlock new configuration
        total_sum = np.sum(radii)
        expansion_factor = 0.006 / (n - 1) + (0.012 / (n - 1)) * 0.7  # Adaptive expansion factor
        
        # Apply controlled expansion to the least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.6  # Over-expansion to unlock
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * 0.8  # Gradual expansion for others
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-9})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())