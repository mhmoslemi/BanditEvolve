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
    constraints = []
    for i in range(n):
        # Left + radius <= 1
        constraints.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        constraints.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        constraints.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        constraints.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints using vectorized broadcasting
    def constraint_func(v, i, j):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        return dx**2 + dy**2 - (v[3*i+2] + v[3*j+2])**2

    # Create overlap constraints with index pairs
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func(v, i, j)})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 1500, "ftol": 1e-12, "eps": 1e-9})
    
    # Apply geometric hashing for spatial reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create geometric hash based on grid structure
        grid_hash = np.random.rand(n, 2) * 0.03
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += grid_hash[i, 0]
            perturbed_v[3*i+1] += grid_hash[i, 1]
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 500, "ftol": 1e-12})

    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute minimum distance for each circle
        min_dists = np.min(dists, axis=1)
        
        # Find underutilized circles (those with smallest distance to others)
        underutilized_indices = np.argsort(min_dists)[:int(n * 0.3)]  # focus on worst-case circles

        # Apply gradient-based expansion to underutilized circles
        expanded_v = v.copy()
        expansion_factor = 0.006 / len(underutilized_indices)  # controlled expansion based on number of underutilized circles
        
        for idx in underutilized_indices:
            expanded_v[3*idx + 2] += expansion_factor * np.random.uniform(1.0, 1.2)
        
        # Re-evaluate with expanded vectors in a tighter loop
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-9})
        
        # Perform edge-case optimization for clusters
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            dists = np.sqrt((centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0])**2 +
                            (centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1])**2)
            
            # Identify tight clusters
            cluster_mask = np.zeros(n, dtype=bool)
            for i in range(n):
                if np.any(dists[i, :] < radii[i] + radii[np.where(dists[i, :] < radii[i] + radii)[0]] - 1e-10):
                    cluster_mask[i] = True
            
            # Adjust radii in clusters with more conservative growth
            if np.any(cluster_mask):
                for i in range(n):
                    if cluster_mask[i]:
                        expanded_v[3*i + 2] = np.clip(radii[i] + (0.003 * np.random.rand()), 1e-6, 0.5)
                    else:
                        expanded_v[3*i + 2] = np.clip(radii[i] + (0.006 * np.random.rand()), 1e-6, 0.5)
            
                # Re-evaluate with modified radii
                res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                               constraints=constraints, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-9})
        
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(radii, 1e-6, 0.5)

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())