import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Randomized geometric clustering initialization
    np.random.seed(42)
    xs = np.random.uniform(0.1, 0.9, n)
    ys = np.random.uniform(0.1, 0.9, n)
    # Group circles into clusters for better expansion potential
    clusters = np.array_split(np.arange(n), 4)
    for cluster in clusters:
        # Cluster center
        cx = np.mean(xs[cluster])
        cy = np.mean(ys[cluster])
        # Spread points around the cluster center
        for i in cluster:
            xs[i] = cx + np.random.normal(0, 0.1)
            ys[i] = cy + np.random.normal(0, 0.1)
    
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
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Select the cluster with the most isolated circles for expansion
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Compute cluster distances
        cluster_distances = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[0][i] - centers[0][j]
                    dy = centers[1][i] - centers[1][j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    cluster_distances[i] += dist
        # Find the cluster with the most isolated circles
        cluster_indices = np.array_split(np.arange(n), 4)
        cluster_scores = [np.mean(cluster_distances[i]) for i in cluster_indices]
        isolated_cluster = cluster_indices[np.argmin(cluster_scores)]
        # Expand the radii of the isolated cluster
        for i in isolated_cluster:
            v[3*i+2] += 0.005
            v[3*i] += 0.01
            v[3*i+1] += 0.01
    
    # Refinement with perturbations
    if res.success:
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())