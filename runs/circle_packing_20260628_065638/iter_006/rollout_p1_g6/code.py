import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a random geometric clustering approach
    np.random.seed(42)
    cluster_centers = np.random.rand(4, 2)
    cluster_centers = (cluster_centers - 0.5) * 2
    cluster_centers = np.clip(cluster_centers, 0, 1)
    
    # Assign each circle to a cluster
    clusters = np.random.randint(0, 4, size=n)
    xs = np.zeros(n)
    ys = np.zeros(n)
    for i in range(n):
        cluster = clusters[i]
        x = cluster_centers[cluster, 0] + np.random.uniform(-0.2, 0.2)
        y = cluster_centers[cluster, 1] + np.random.uniform(-0.2, 0.2)
        xs[i] = np.clip(x, 0, 1)
        ys[i] = np.clip(y, 0, 1)
    
    r0 = 0.15
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
    
    # Non-local reconfiguration strategy: expand the most isolated cluster
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Find the cluster with the largest minimum distance to other clusters
        cluster_distances = np.zeros(4)
        for i in range(n):
            cluster = clusters[i]
            for j in range(n):
                if clusters[j] != cluster:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    cluster_distances[cluster] = np.max([cluster_distances[cluster], dist])
        
        # Expand the most isolated cluster
        isolated_cluster = np.argmax(cluster_distances)
        perturb_indices = np.where(clusters == isolated_cluster)[0]
        
        # Apply controlled expansion to the radii of the isolated cluster
        expansion_factor = 1.2
        perturbed_v = v.copy()
        for i in perturb_indices:
            perturbed_v[3*i+2] = np.clip(radii[i] * expansion_factor, 1e-4, 0.5)
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())