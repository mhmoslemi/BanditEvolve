import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering
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
        # Stagger alternate rows to reduce overlap
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
    
    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Randomized geometric hashing scheme: reconfigure based on spatial hashing
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Create a hash grid for spatial hashing
        grid_size = 0.1
        hash_grid = {}
        for i in range(n):
            x = centers[0][i]
            y = centers[1][i]
            grid_x = int(x / grid_size)
            grid_y = int(y / grid_size)
            key = (grid_x, grid_y)
            if key not in hash_grid:
                hash_grid[key] = []
            hash_grid[key].append(i)
        # Identify clusters and perturb them to trigger new configurations
        cluster_indices = []
        for key in hash_grid:
            if len(hash_grid[key]) > 1:
                for idx in hash_grid[key]:
                    cluster_indices.append(idx)
        # Randomly perturb cluster indices
        if cluster_indices:
            cluster_indices = np.unique(np.random.choice(cluster_indices, size=min(10, len(cluster_indices)), replace=False))
            perturbation = np.random.rand(len(cluster_indices) * 3) * 0.05
            perturbed_v = v.copy()
            idx = 0
            for i in cluster_indices:
                perturbed_v[3*i] += perturbation[idx]
                perturbed_v[3*i+1] += perturbation[idx+1]
                perturbed_v[3*i+2] += perturbation[idx+2]
                idx += 3
            # Re-evaluate with perturbed parameters
            res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion: expand the circle with the smallest non-zero radius
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Find the circle with the smallest non-zero radius
        valid_radii = radii[radii > 1e-6]
        if len(valid_radii) > 0:
            min_radius_idx = np.argmin(valid_radii)
            expanded_idx = np.where(radii == valid_radii[min_radius_idx])[0][0]
            # Expand radius slightly and adjust its position to maintain feasibility
            v[3*expanded_idx + 2] += 0.002
            v[3*expanded_idx] += 0.005
            v[3*expanded_idx+1] += 0.005
            # Re-evaluate with adjusted parameters
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())