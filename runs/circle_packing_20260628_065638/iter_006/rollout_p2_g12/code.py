import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized clustering algorithm
    def cluster_positions(n, cols, rows):
        cluster_centers = []
        for i in range(cols):
            for j in range(rows):
                # Generate a cluster center with small random offsets
                x_center = (i + 0.5) / cols + np.random.uniform(-0.05, 0.05)
                y_center = (j + 0.5) / rows + np.random.uniform(-0.05, 0.05)
                cluster_centers.append((x_center, y_center))
        # Randomly assign circles to clusters
        cluster_assignments = np.random.choice(cols * rows, n)
        xs = []
        ys = []
        for i in range(n):
            cluster_idx = cluster_assignments[i]
            cluster_x, cluster_y = cluster_centers[cluster_idx]
            # Add slight perturbation to avoid perfect symmetry
            perturbation = np.random.uniform(-0.05, 0.05, size=2)
            xs.append(cluster_x + perturbation[0])
            ys.append(cluster_y + perturbation[1])
        return np.array(xs), np.array(ys)
    
    xs, ys = cluster_positions(n, cols, rows)
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
    
    # Non-local reconfiguration: identify the most isolated cluster and expand it
    if res.success:
        v = res.x
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        # Compute distances from each circle to the nearest other circle
        dists = np.zeros(n)
        for i in range(n):
            min_dist = np.inf
            for j in range(n):
                if i != j:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < min_dist:
                        min_dist = dist
            dists[i] = min_dist
        # Select the cluster with the largest minimum distance
        isolated_indices = np.argsort(dists)[-5:]  # Select top 5 most isolated circles
        # Increase radii of these circles by 5% and re-optimize
        perturbed_v = v.copy()
        for i in isolated_indices:
            perturbed_v[3*i+2] *= 1.05
            perturbed_v[3*i+2] = np.clip(perturbed_v[3*i+2], 1e-4, 0.5)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())