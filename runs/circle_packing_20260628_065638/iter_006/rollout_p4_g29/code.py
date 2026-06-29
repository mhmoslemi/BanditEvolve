import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a random geometric clustering algorithm
    xs = np.random.uniform(0.05, 0.95, n)
    ys = np.random.uniform(0.05, 0.95, n)
    # Group circles into clusters for better expansion
    cluster_centers = np.random.uniform(0.2, 0.8, (cols, 2))
    cluster_radii = np.random.uniform(0.05, 0.15, cols)
    # Assign each circle to a cluster
    cluster_indices = np.argmin(np.sum((xs[:, np.newaxis] - cluster_centers[np.newaxis, :, 0])**2 + 
                                       (ys[:, np.newaxis] - cluster_centers[np.newaxis, :, 1])**2, axis=1))
    # Adjust positions to cluster around centers
    xs = cluster_centers[cluster_indices, 0] + np.random.normal(0, 0.05, n)
    ys = cluster_centers[cluster_indices, 1] + np.random.normal(0, 0.05, n)
    # Ensure circles are within bounds and not overlapping
    for i in range(n):
        xs[i] = np.clip(xs[i], 0.05, 0.95)
        ys[i] = np.clip(ys[i], 0.05, 0.95)
    # Calculate initial radii based on cluster size
    cluster_sizes = np.bincount(cluster_indices)
    r0 = np.clip(0.3 / np.sqrt(cluster_sizes[cluster_indices]) - 1e-3, 1e-4, 0.5)
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

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
    
    # Targeted reconfiguration: expand the most isolated cluster
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Calculate isolation metric for clusters
        cluster_isolation = np.zeros(cols)
        for i in range(n):
            cluster_idx = cluster_indices[i]
            dists = np.zeros(cols)
            for j in range(cols):
                dx = centers[0][i] - cluster_centers[j, 0]
                dy = centers[1][i] - cluster_centers[j, 1]
                dists[j] = np.sqrt(dx*dx + dy*dy)
            cluster_isolation[cluster_idx] += np.min(dists)
        isolated_cluster = np.argmax(cluster_isolation)
        # Expand radii of the most isolated cluster
        for i in range(n):
            if cluster_indices[i] == isolated_cluster:
                v[3*i+2] += 0.005
                v[3*i] += np.random.uniform(-0.01, 0.01)
                v[3*i+1] += np.random.uniform(-0.01, 0.01)
        # Clip radii to ensure they stay within bounds
        v[2::3] = np.clip(v[2::3], 1e-4, 0.5)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())