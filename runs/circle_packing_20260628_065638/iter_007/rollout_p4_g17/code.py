import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using randomized geometric clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Random clustering with controlled perturbation
        cluster_id = np.random.randint(0, 2)
        if cluster_id == 0:
            x += np.random.uniform(-0.05 / cols, 0.05 / cols)
            y += np.random.uniform(-0.05 / rows, 0.05 / rows)
        else:
            x += np.random.uniform(0.05 / cols, 0.1 / cols)
            y += np.random.uniform(0.05 / rows, 0.1 / rows)
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
    def vectorized_overlap_constraints(v):
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        dx = centers[0][:, None] - centers[0][:]
        dy = centers[1][:, None] - centers[1][:]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (radii[:, None] + radii[None, :])**2
        return dist_sq - min_dist_sq

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized constraints for all pairs
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Radical reconfiguration: identify the tightest cluster and expand it
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Compute pairwise distances and group by clusters
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[0][i] - centers[0][j]
                    dy = centers[1][i] - centers[1][j]
                    dists[i][j] = np.sqrt(dx*dx + dy*dy)
        # Find the cluster with the smallest average distance
        cluster_indices = np.arange(n)
        np.random.shuffle(cluster_indices)
        clusters = []
        cluster_size = n // 2
        for i in range(2):
            cluster = cluster_indices[i*cluster_size:(i+1)*cluster_size]
            clusters.append(cluster)
        # Expand the cluster with smaller average distance
        avg_dists = []
        for cluster in clusters:
            cluster_dists = dists[np.ix_(cluster, cluster)]
            avg_dist = np.mean(cluster_dists[cluster_dists > 0])
            avg_dists.append(avg_dist)
        expand_cluster = clusters[np.argmin(avg_dists)]
        # Increase radii of the cluster members
        expand_indices = np.array(expand_cluster)
        v[3*expand_indices + 2] += 0.002
        # Re-evaluate with modified parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())