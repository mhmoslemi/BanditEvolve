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
        x = x_center + np.random.uniform(-0.04, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.04)
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
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing and adaptive tolerance
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_radius = np.min([v[3*i+2], v[3*j+2]])
                return dist_sq - (2*min_radius + 1e-8)**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tighter tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "eps": 1e-12})
    
    # Trigger major geometric shift with randomized geometric hashing
    if res.success:
        v = res.x
        # Create a random geometric hash map to induce topological change
        hash_map = np.random.rand(n, 2) * 0.1
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += hash_map[i, 0]
            perturbed_v[3*i+1] += hash_map[i, 1]
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12})
    
    # Targeted radius expansion on smallest non-zero radius with adjacency constraints
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate distances to find the smallest radius and its adjacent circles
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Identify smallest radius and its adjacent circles
        min_radius_idx = np.argmin(radii)
        min_dist = np.min(dists[min_radius_idx, :])
        min_dist_idx = np.argmin(dists[min_radius_idx, :])
        
        # Expand the smallest radius while maintaining adjacency constraints
        total_sum = np.sum(radii)
        target_total_sum = total_sum + 0.005
        expansion = (target_total_sum - total_sum) / (n - 1)
        
        # Create new radius vector with expansion
        new_radii = radii.copy()
        new_radii[min_radius_idx] += expansion * 1.2  # Slight over-expansion to trigger reconfiguration
        for i in range(n):
            if i != min_radius_idx:
                new_radii[i] += expansion
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and maintain adjacency constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())