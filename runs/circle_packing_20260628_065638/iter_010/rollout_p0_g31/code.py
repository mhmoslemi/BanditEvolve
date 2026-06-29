import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and asymmetry
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Randomized offset with row-dependent asymmetry
        x += np.random.uniform(-0.05, 0.05)
        y += np.random.uniform(-0.05, 0.05)
        # Alternate row staggering for asymmetry
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
    
    # Radically reconfigure by replacing spatial arrangement with randomized geometric clustering
    if res.success:
        v = res.x
        # Randomized geometric clustering: place circles in a hexagonal grid
        cluster_centers = np.random.rand(4, 2)
        cluster_centers = np.vstack((cluster_centers, np.random.rand(4, 2)))
        cluster_radii = np.random.uniform(0.05, 0.1, 8)
        cluster_positions = []
        for i in range(8):
            for j in range(8):
                if i < 4 and j < 4:
                    x = cluster_centers[i, 0] + 0.25 * (j % 2)
                    y = cluster_centers[i, 1] + 0.25 * (j // 2)
                    cluster_positions.append([x, y])
        # Distribute remaining circles in a uniform grid
        for i in range(8, 26):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            cluster_positions.append([x, y])
        # Scale and adjust positions to fit in the unit square
        xs_new = np.array(cluster_positions)[:, 0]
        ys_new = np.array(cluster_positions)[:, 1]
        xs_new = xs_new * 0.9 + np.random.uniform(-0.05, 0.05, n)
        ys_new = ys_new * 0.9 + np.random.uniform(-0.05, 0.05, n)
        # Initialize radii with a small perturbation
        r0_new = 0.35 / np.sqrt(n) - 1e-3
        r0_new = np.clip(r0_new + np.random.uniform(-0.005, 0.005, n), 1e-4, 0.5)
        v_new = np.empty(3 * n)
        v_new[0::3] = xs_new
        v_new[1::3] = ys_new
        v_new[2::3] = r0_new
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion: expand the most tightly packed cluster
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Calculate pairwise distances between circles
        distances = np.zeros(n * n)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[0][i] - centers[0][j]
                dy = centers[1][i] - centers[1][j]
                distances[i * n + j] = np.sqrt(dx * dx + dy * dy)
                distances[j * n + i] = distances[i * n + j]
        # Identify the cluster with the smallest average distance
        cluster_indices = np.arange(n)
        cluster_distances = np.zeros(n)
        for i in range(n):
            cluster_distances[i] = np.mean(distances[i * n + cluster_indices[cluster_indices != i]])
        cluster_idx = np.argmin(cluster_distances)
        # Expand radii of cluster members slightly
        for i in range(n):
            if np.linalg.norm([centers[0][i] - centers[0][cluster_idx], centers[1][i] - centers[1][cluster_idx]]) < 0.3:
                v[3*i + 2] += 0.005
                v[3*i] += 0.01
                v[3*i + 1] += 0.01
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())