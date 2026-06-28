import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / cols
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

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

    def vectorized_overlap_constraint(v):
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_radii = v[2::3]
        dx = x_centers[:, np.newaxis] - x_centers[np.newaxis, :]
        dy = y_centers[:, np.newaxis] - y_centers[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (r_radii[:, np.newaxis] + r_radii[np.newaxis, :])**2
        return dist_sq - min_dist_sq

    def constraint_func(v, i, j):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2

    overlap_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            overlap_cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func(v, i, j)})

    cons.extend(overlap_cons)

    def random_partition(v):
        # Partition the circles into independent subcomponents
        cluster_size = 6
        clusters = []
        for i in range(0, n, cluster_size):
            cluster = v[3*i:3*(i+cluster_size)]
            clusters.append(cluster)
        return clusters

    def permute_clusters(clusters):
        # Apply a global permutation to the clusters
        np.random.shuffle(clusters)
        return np.concatenate(clusters)

    def enforce_radius_growth(v, clusters, target_growth=0.05):
        # Calculate average radius for each cluster
        cluster_radii = []
        for cluster in clusters:
            r = cluster[2::3]
            avg_r = np.mean(r)
            cluster_radii.append(avg_r)
        # Identify clusters that can grow
        growable_clusters = [i for i in range(len(clusters)) if cluster_radii[i] < 0.2]
        if len(growable_clusters) > 0:
            # Randomly select one cluster to grow
            selected = np.random.choice(growable_clusters)
            # Perturb the positions of the cluster to allow growth
            cluster = clusters[selected]
            perturbation = np.random.rand(3 * len(cluster)) * 0.1
            cluster_perturb = cluster + perturbation
            cluster_perturb[0::3] = np.clip(cluster_perturb[0::3], 0.0, 1.0)
            cluster_perturb[1::3] = np.clip(cluster_perturb[1::3], 0.0, 1.0)
            cluster_perturb[2::3] = np.clip(cluster_perturb[2::3], 1e-4, 0.5)
            clusters[selected] = cluster_perturb
        return clusters

    # Apply the forced topological reconfiguration
    v_partitions = random_partition(v0)
    v_permuted = permute_clusters(v_partitions)
    v_enforced = enforce_radius_growth(v_permuted, v_partitions)
    v_reconfigured = np.concatenate(v_enforced)

    # Run the optimization with reconfigured initial guess
    res = minimize(neg_sum_radii, v_reconfigured, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())