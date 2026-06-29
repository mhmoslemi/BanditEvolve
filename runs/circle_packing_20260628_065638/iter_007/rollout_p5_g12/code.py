import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Randomized geometric clustering initialization
    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 1, (n, 2))
    # Cluster points into groups and spread them to avoid overlap
    clusters = np.array_split(positions, cols)
    for cluster in clusters:
        cluster += np.random.uniform(-0.1, 0.1, cluster.shape)
    positions = np.vstack(clusters)
    
    # Initial radii based on cluster density
    r0 = 0.25 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = positions[:, 0]
    v0[1::3] = positions[:, 1]
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

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Local refinement: expand the most tightly packed cluster
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Compute pairwise distances
        dists = np.zeros(n)
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[0][i] - centers[0][j]
                dy = centers[1][i] - centers[1][j]
                dists[i] += np.sqrt(dx*dx + dy*dy)
                dists[j] += np.sqrt(dx*dx + dy*dy)
        # Find the cluster with the smallest average distance
        cluster_size = 5
        cluster_indices = np.array_split(np.arange(n), n // cluster_size)
        cluster_min = np.inf
        cluster_idx = 0
        for i, indices in enumerate(cluster_indices):
            avg_dist = np.mean(dists[indices])
            if avg_dist < cluster_min:
                cluster_min = avg_dist
                cluster_idx = i
        # Expand the selected cluster
        cluster = cluster_indices[cluster_idx]
        # Increase radii of cluster members by a small amount
        v[3*cluster + 2] += 0.005
        # Perturb positions slightly
        v[3*cluster + 0] += np.random.uniform(-0.01, 0.01, len(cluster))
        v[3*cluster + 1] += np.random.uniform(-0.01, 0.01, len(cluster))
        # Re-optimization
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())