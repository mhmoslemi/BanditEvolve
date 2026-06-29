import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    xs = np.random.rand(n)
    ys = np.random.rand(n)
    # Group circles into 5 clusters for geometric clustering
    cluster_ids = np.random.choice(5, size=n)
    cluster_centers = np.random.rand(5, 2)
    cluster_centers[:, 0] *= 1.0
    cluster_centers[:, 1] *= 1.0
    # Assign each circle to a cluster and position it near the cluster center
    for i in range(n):
        cid = cluster_ids[i]
        xs[i] = cluster_centers[cid, 0] + np.random.uniform(-0.1, 0.1)
        ys[i] = cluster_centers[cid, 1] + np.random.uniform(-0.1, 0.1)
        # Ensure circles are within the unit square
        xs[i] = np.clip(xs[i], 0.0, 1.0)
        ys[i] = np.clip(ys[i], 0.0, 1.0)
    
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

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Apply controlled radius expansion to the most isolated cluster
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Identify the most isolated cluster using distance to other circles
        distances = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    distances[i] += dist
        # Normalize and find the cluster with the highest isolation score
        isolation_scores = distances / np.sum(distances)
        cluster_ids = np.array(cluster_ids, dtype=int)
        cluster_isolation = np.zeros(5)
        for i in range(n):
            cid = cluster_ids[i]
            cluster_isolation[cid] += isolation_scores[i]
        # Select the cluster with the highest isolation score
        most_isolated_cluster = np.argmax(cluster_isolation)
        # Expand radii for the most isolated cluster
        expansion_factor = 1.2
        # Select circles in the most isolated cluster
        cluster_indices = np.where(cluster_ids == most_isolated_cluster)[0]
        # Calculate distance from cluster members to cluster boundary
        cluster_centers = np.zeros((5, 2))
        for i in range(5):
            cx = 0.0
            cy = 0.0
            count = 0
            for j in range(n):
                if cluster_ids[j] == i:
                    cx += centers[j, 0]
                    cy += centers[j, 1]
                    count += 1
            cluster_centers[i, 0] = cx / count
            cluster_centers[i, 1] = cy / count
        # Calculate distance from cluster members to cluster boundary
        for i in cluster_indices:
            dx = centers[i, 0] - cluster_centers[most_isolated_cluster, 0]
            dy = centers[i, 1] - cluster_centers[most_isolated_cluster, 1]
            dist_to_boundary = np.sqrt(dx*dx + dy*dy)
            # Scale expansion to avoid overlapping with boundary
            expansion = min(expansion_factor * radii[i], dist_to_boundary - radii[i])
            v[3*i + 2] += expansion
        # Clip radii to ensure they stay within bounds
        v[2::3] = np.clip(v[2::3], 1e-4, 0.5)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())