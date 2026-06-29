import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    # Create clusters by grouping nearby indices and placing them in random positions
    cluster_size = 5
    clusters = []
    for i in range(n // cluster_size + 1):
        cluster = np.random.rand(cluster_size, 2)
        cluster[:, 0] *= 0.4
        cluster[:, 1] *= 0.4
        cluster += np.random.rand(cluster_size, 2) * 0.3
        clusters.append(cluster)
    clusters = np.vstack(clusters)
    # Add remaining circles as singletons
    for i in range(len(clusters), n):
        clusters = np.vstack([clusters, np.random.rand(2) * 0.5])
    xs = clusters[:, 0]
    ys = clusters[:, 1]
    
    r0 = 0.25 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

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
    
    # Controlled radius expansion for the most isolated cluster
    if res.success:
        v = res.x
        radii = v[2::3]
        # Compute isolation metric as inverse of average distance to other circles
        dists = np.zeros(n)
        for i in range(n):
            dists[i] = np.mean(np.sqrt(np.sum((v[0::3] - v[3*i] ** 2 + v[1::3] - v[3*i+1]) ** 2)))
        # Select the most isolated cluster (group of 5 circles)
        cluster_indices = np.array_split(np.arange(n), n // cluster_size + 1)[0]
        isolation = np.array([np.mean(dists[i] for i in cluster_indices)])
        # Perturb the cluster and expand radii
        perturb_indices = cluster_indices
        # Apply small random perturbation to positions
        perturbation = 0.05 * np.random.rand(len(perturb_indices) * 3)
        perturbed_v = v.copy()
        idx = 0
        for i in perturb_indices:
            perturbed_v[3*i] += perturbation[idx]
            perturbed_v[3*i+1] += perturbation[idx+1]
            perturbed_v[3*i+2] += perturbation[idx+2]
            idx += 3
        # Clip radii to ensure they stay within bounds
        perturbed_v[2::3] = np.clip(perturbed_v[2::3], 1e-4, 0.5)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())