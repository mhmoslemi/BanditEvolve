import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    xs = np.random.rand(n)
    ys = np.random.rand(n)
    # Group circles into clusters and place them in a grid-like arrangement
    clusters = np.random.choice(np.arange(cols), size=n, p=np.ones(cols)/cols)
    cluster_centers = np.zeros((cols, 2))
    for i in range(cols):
        cluster_centers[i, 0] = (i + 0.5) / cols
        cluster_centers[i, 1] = 0.5
    # Assign cluster coordinates to each circle
    for i in range(n):
        cluster_idx = clusters[i]
        cx, cy = cluster_centers[cluster_idx]
        # Randomly perturb within the cluster to avoid perfect symmetry
        xs[i] = cx + np.random.uniform(-0.1, 0.1)
        ys[i] = cy + np.random.uniform(-0.1, 0.1)
    # Ensure all points are within bounds
    xs = np.clip(xs, 0.0, 1.0)
    ys = np.clip(ys, 0.0, 1.0)
    
    r0 = 0.3 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
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
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Apply controlled radius expansion to the most isolated cluster
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Calculate isolation metric for each cluster
        cluster_isolation = np.zeros(cols)
        for c in range(cols):
            cluster_indices = np.where(clusters == c)[0]
            if len(cluster_indices) == 0:
                continue
            isolation = 0.0
            for i in cluster_indices:
                dist = np.inf
                for j in range(n):
                    if i == j:
                        continue
                    dx = centers[0][i] - centers[0][j]
                    dy = centers[1][i] - centers[1][j]
                    dist = min(dist, np.sqrt(dx*dx + dy*dy))
                isolation += dist
            cluster_isolation[c] = isolation
        # Find the most isolated cluster and expand its radii
        isolated_cluster = np.argmin(cluster_isolation)
        isolated_indices = np.where(clusters == isolated_cluster)[0]
        if len(isolated_indices) > 0:
            for i in isolated_indices:
                v[3*i + 2] += 0.002
                v[3*i] += 0.005
                v[3*i + 1] += 0.005
            # Re-evaluate with perturbed parameters
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())