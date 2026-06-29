import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    # Create a grid of cluster centers and perturb them
    cluster_centers = []
    for i in range(rows):
        for j in range(cols):
            # Create cluster center with randomized offset
            x = (j + 0.5) / cols + np.random.uniform(-0.1, 0.1)
            y = (i + 0.5) / rows + np.random.uniform(-0.1, 0.1)
            # Ensure cluster is within bounds
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            cluster_centers.append((x, y))
    
    # Assign circles to clusters and initialize positions
    xs = []
    ys = []
    for i in range(n):
        cluster_idx = i % rows
        x = cluster_centers[cluster_idx][0] + np.random.uniform(-0.1, 0.1)
        y = cluster_centers[cluster_idx][1] + np.random.uniform(-0.1, 0.1)
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        xs.append(x)
        ys.append(y)
    
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
    
    # Force non-local reconfiguration by expanding the most isolated cluster
    if res.success:
        v = res.x
        radii = v[2::3]
        # Identify the cluster with the largest minimum distance to other circles
        min_dist = np.full(n, np.inf)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist[i] = np.minimum(min_dist[i], dist)
        
        # Select the cluster with the largest minimum distance (most isolated)
        isolated_indices = np.argsort(min_dist)[-3:]
        # Increase radii of circles in the isolated cluster by a controlled amount
        perturbation = 0.01 * np.random.rand(len(isolated_indices))
        perturbed_v = v.copy()
        for i in isolated_indices:
            perturbed_v[3*i+2] += perturbation[np.where(isolated_indices == i)[0][0]]
            # Ensure radii stay within bounds
            perturbed_v[3*i+2] = np.clip(perturbed_v[3*i+2], 1e-4, 0.5)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())