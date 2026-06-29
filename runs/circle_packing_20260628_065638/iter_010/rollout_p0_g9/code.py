import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Randomized geometric clustering initialization
    xs = np.random.rand(n) * 0.9 + 0.05
    ys = np.random.rand(n) * 0.9 + 0.05
    # Cluster centers with controlled spacing
    cluster_centers = np.array([(i % cols + 0.5) / cols, (i // cols + 0.5) / rows] for i in range(n))
    # Assign each circle to a cluster with randomized offset
    cluster_assignments = np.random.randint(0, cols, n)
    for i in range(n):
        cluster_idx = cluster_assignments[i]
        x = cluster_centers[cluster_idx, 0] + np.random.uniform(-0.1, 0.1)
        y = cluster_centers[cluster_idx, 1] + np.random.uniform(-0.1, 0.1)
        xs[i] = np.clip(x, 0.0, 1.0)
        ys[i] = np.clip(y, 0.0, 1.0)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints
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
    
    # Radical reconfiguration: trigger new geometric clustering
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Randomized geometric clustering with new cluster assignments
        cluster_assignments = np.random.randint(0, cols, n)
        for i in range(n):
            cluster_idx = cluster_assignments[i]
            x = cluster_centers[cluster_idx, 0] + np.random.uniform(-0.1, 0.1)
            y = cluster_centers[cluster_idx, 1] + np.random.uniform(-0.1, 0.1)
            v[3*i] = np.clip(x, 0.0, 1.0)
            v[3*i+1] = np.clip(y, 0.0, 1.0)
        # Re-evaluate with new cluster positions
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion: expand the most tightly packed cluster
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Calculate cluster tightness based on distance between circles
        cluster_tightness = np.zeros(cols)
        for i in range(n):
            cluster_idx = cluster_assignments[i]
            cluster_tightness[cluster_idx] += np.sqrt((centers[0][i] - np.mean(centers[0][cluster_assignments == cluster_idx]))**2 +
                                                      (centers[1][i] - np.mean(centers[1][cluster_assignments == cluster_idx]))**2)
        # Identify the most tightly packed cluster
        most_tight_cluster = np.argmin(cluster_tightness)
        # Expand the cluster's radii and adjust positions to maintain feasibility
        for i in range(n):
            if cluster_assignments[i] == most_tight_cluster:
                v[3*i + 2] += 0.005
                v[3*i] += 0.005
                v[3*i+1] += 0.005
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())