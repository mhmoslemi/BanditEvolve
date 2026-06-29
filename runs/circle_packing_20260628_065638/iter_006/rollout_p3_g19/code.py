import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    xs = np.random.rand(n)
    ys = np.random.rand(n)
    radii = np.full(n, 0.3 / cols - 1e-3)
    
    # Cluster centers using k-means
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5, random_state=0).fit(np.column_stack([xs, ys]))
    cluster_centers = kmeans.cluster_centers_
    
    # Assign each circle to a cluster
    cluster_indices = kmeans.labels_
    
    # Distribute circles within clusters
    for i in range(n):
        cluster = cluster_indices[i]
        dx = np.random.rand() * (cluster_centers[cluster, 0] - 0.3) + 0.15
        dy = np.random.rand() * (cluster_centers[cluster, 1] - 0.3) + 0.15
        xs[i] = cluster_centers[cluster, 0] - dx
        ys[i] = cluster_centers[cluster, 1] - dy
    
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = radii

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized overlap constraints
    def vectorized_overlap(v):
        radii = v[2::3]
        x = v[0::3]
        y = v[1::3]
        dist_sq = np.zeros((n, n))
        for i in range(n):
            dx = x - x[i]
            dy = y - y[i]
            dist_sq[:, i] = dx*dx + dy*dy
        return dist_sq - (radii[:, np.newaxis] + radii[np.newaxis, :]) ** 2

    # Create constraints for all pairs using vectorized calculation
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Create constraints for all pairs
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: vectorized_overlap(v)[i, j]})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Non-local reconfiguration: find the most isolated cluster and expand it
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute isolation metric: distance to nearest neighbor
        isolation = np.zeros(n)
        for i in range(n):
            min_dist = np.inf
            for j in range(n):
                if i != j:
                    dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                    min_dist = min(min_dist, dist)
            isolation[i] = min_dist
        
        # Select the most isolated cluster
        isolated_indices = np.argsort(isolation)[-3:]
        
        # Apply controlled expansion to the most isolated cluster
        perturbed_v = v.copy()
        for i in isolated_indices:
            perturbed_v[3*i+2] += 0.01  # Increase radius slightly
    
        # Clip radii to ensure they stay within bounds
        perturbed_v[2::3] = np.clip(perturbed_v[2::3], 1e-4, 0.5)
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())