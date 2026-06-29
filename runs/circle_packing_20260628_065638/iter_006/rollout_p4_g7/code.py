import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a geometric clustering algorithm
    xs = np.random.rand(n)
    ys = np.random.rand(n)
    # Group circles into clusters with a grid-based spatial clustering
    cluster_centers = np.zeros((cols, rows, 2))
    for i in range(cols):
        for j in range(rows):
            cluster_centers[i, j, 0] = i / (cols - 1)
            cluster_centers[i, j, 1] = j / (rows - 1)
    # Assign circles to clusters
    cluster_assignments = np.random.randint(0, cols * rows, size=n)
    xs = np.zeros(n)
    ys = np.zeros(n)
    for i in range(n):
        cluster_idx = cluster_assignments[i]
        cluster_row = cluster_idx // cols
        cluster_col = cluster_idx % cols
        xs[i] = cluster_centers[cluster_col, cluster_row, 0] + np.random.uniform(-0.05, 0.05)
        ys[i] = cluster_centers[cluster_col, cluster_row, 1] + np.random.uniform(-0.05, 0.05)
    
    r0 = 0.3 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
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
    
    # Apply controlled radius expansion to the most isolated cluster
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Compute pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[0][i] - centers[0][j]
                    dy = centers[1][i] - centers[1][j]
                    dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Compute average distance from each circle to all others
        avg_dists = np.mean(dists, axis=1)
        # Identify the cluster with the most isolated circles
        cluster_indices = np.zeros(n, dtype=int)
        for i in range(cols * rows):
            row = i // cols
            col = i % cols
            cluster_center = np.array([col / (cols - 1), row / (rows - 1)])
            distances = np.sqrt((centers[0] - cluster_center[0])**2 + (centers[1] - cluster_center[1])**2)
            cluster_indices[np.argsort(distances)[:n//2]] = i
        # Identify the most isolated cluster
        cluster_avg_dists = np.zeros(cols * rows)
        for i in range(cols * rows):
            cluster_avg_dists[i] = np.mean(avg_dists[cluster_indices == i])
        isolated_cluster = np.argmin(cluster_avg_dists)
        # Select circles in the isolated cluster
        cluster_circles = np.where(cluster_indices == isolated_cluster)[0]
        # Apply controlled radius expansion
        for i in cluster_circles:
            v[3*i+2] += 0.002
            v[3*i] += 0.005
            v[3*i+1] += 0.005
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())