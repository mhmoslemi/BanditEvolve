import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    xs = []
    ys = []
    # Create a grid of cluster centers
    for i in range(cols):
        for j in range(rows):
            x = (i + 0.5) / cols
            y = (j + 0.5) / rows
            # Add random perturbation to cluster centers
            x += np.random.uniform(-0.05, 0.05)
            y += np.random.uniform(-0.05, 0.05)
            xs.append(x)
            ys.append(y)
    # Randomly assign circles to clusters
    cluster_assignments = np.random.choice(cols * rows, n)
    xs = np.array(xs)[cluster_assignments]
    ys = np.array(ys)[cluster_assignments]
    
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
    
    # Non-local reconfiguration: identify the most isolated cluster and expand its radius
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Calculate distances for each circle to all other circles
        dists = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[0][i] - centers[0][j]
                    dy = centers[1][i] - centers[1][j]
                    dists[i] += np.sqrt(dx*dx + dy*dy)
        # Find the cluster with the most isolated circles
        cluster_indices = np.array([i // rows for i in cluster_assignments])
        cluster_isolation = np.zeros(cols)
        for i in range(n):
            cluster_isolation[cluster_indices[i]] += dists[i]
        isolated_cluster = np.argmin(cluster_isolation)
        # Select all circles in the isolated cluster
        cluster_circles = np.where(cluster_indices == isolated_cluster)[0]
        # Expand the radius of the most isolated circle in this cluster
        if len(cluster_circles) > 0:
            circle_dists = dists[cluster_circles]
            isolated_circle = cluster_circles[np.argmin(circle_dists)]
            v[3*isolated_circle + 2] += 0.002
            v[3*isolated_circle + 0] += 0.005
            v[3*isolated_circle + 1] += 0.005
            # Re-evaluate with perturbed parameters
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())