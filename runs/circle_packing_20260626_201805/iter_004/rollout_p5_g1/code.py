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

    overlap_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            overlap_cons.append({"type": "ineq", "fun": constraint_func})

    cons.extend(overlap_cons)

    def topological_reconfiguration(v):
        # Generate a list of indices for the current configuration
        indices = np.arange(n)
        # Randomly shuffle the indices
        np.random.shuffle(indices)
        # Create a new configuration by grouping circles into clusters
        cluster_size = 3
        cluster_centers = []
        cluster_radii = []
        for i in range(0, n, cluster_size):
            cluster_indices = indices[i:i+cluster_size]
            # Get the positions and radii of the cluster
            cluster_x = v[3*cluster_indices]
            cluster_y = v[3*cluster_indices + 1]
            cluster_r = v[3*cluster_indices + 2]
            # Calculate the centroid of the cluster
            centroid_x = np.mean(cluster_x)
            centroid_y = np.mean(cluster_y)
            # Compute the minimum distance between cluster members
            min_dist = np.min(np.sqrt((cluster_x[:, np.newaxis] - cluster_x[np.newaxis, :])**2 + 
                                     (cluster_y[:, np.newaxis] - cluster_y[np.newaxis, :])**2))
            # Compute the minimum radius that can be assigned to the cluster
            min_r = min_dist / 2
            # Assign a new radius to the cluster
            new_r = np.clip(min_r, 1e-4, 0.5)
            # Store the new cluster configuration
            cluster_centers.append([centroid_x, centroid_y])
            cluster_radii.append(new_r)
        # Create a new configuration with the cluster centers and radii
        new_v = np.zeros(3 * n)
        for i in range(n):
            new_v[3*i] = cluster_centers[i][0]
            new_v[3*i + 1] = cluster_centers[i][1]
            new_v[3*i + 2] = cluster_radii[i]
        return new_v

    # Initial optimization with topological reconfiguration
    v_reconfigured = topological_reconfiguration(v0)
    res = minimize(neg_sum_radii, v_reconfigured, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0

    # Final optimization
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-9})
    v = res.x if res.success else v0

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())