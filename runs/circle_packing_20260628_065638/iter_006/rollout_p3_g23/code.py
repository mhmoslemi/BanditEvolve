import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a geometric clustering algorithm
    xs = np.random.rand(n)
    ys = np.random.rand(n)
    # Group circles into clusters with random cluster centers
    cluster_centers = np.random.rand(5, 2)
    cluster_centers = np.clip(cluster_centers, 0.1, 0.9)
    # Assign circles to clusters based on distance
    cluster_assignments = np.argmin(np.linalg.norm(cluster_centers[:, np.newaxis] - np.column_stack((xs, ys)), axis=2), axis=0)
    # Adjust positions within clusters for better packing
    for cluster_id in range(5):
        cluster_mask = (cluster_assignments == cluster_id)
        cluster_x = xs[cluster_mask]
        cluster_y = ys[cluster_mask]
        # Center cluster around its mean
        mean_x = np.mean(cluster_x)
        mean_y = np.mean(cluster_y)
        xs[cluster_mask] = cluster_x - mean_x + 0.5
        ys[cluster_mask] = cluster_y - mean_y + 0.5
    # Clip positions to unit square
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
    
    # Non-local reconfiguration: expand the most isolated cluster
    if res.success:
        v = res.x
        radii = v[2::3]
        # Calculate isolation score based on minimum distance to other circles
        isolation_scores = np.zeros(n)
        for i in range(n):
            min_dist = np.inf
            for j in range(n):
                if i != j:
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < min_dist:
                        min_dist = dist
            isolation_scores[i] = min_dist - (radii[i] + radii[j]) if j < n else min_dist - radii[i]
        # Find the cluster with the largest isolation score
        cluster_indices = np.argsort(isolation_scores)[-3:]
        # Apply controlled expansion to the most isolated cluster
        perturbation = 0.05 * np.random.rand(len(cluster_indices) * 3)
        perturbed_v = v.copy()
        idx = 0
        for i in cluster_indices:
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