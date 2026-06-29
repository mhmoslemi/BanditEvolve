import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Randomized geometric clustering initialization
    np.random.seed(42)
    cluster_centers = np.random.rand(n, 2)
    cluster_centers = (cluster_centers - 0.5) * 0.8 + 0.5  # Center and scale
    cluster_radii = np.full(n, 0.05)
    
    # Assign each circle to a cluster
    clusters = np.random.randint(0, cols, n)
    cluster_positions = [[] for _ in range(cols)]
    cluster_radii_list = [[] for _ in range(cols)]
    for i in range(n):
        cluster = clusters[i]
        cluster_positions[cluster].append(cluster_centers[i])
        cluster_radii_list[cluster].append(cluster_radii[i])
    
    # Compute initial positions
    xs = []
    ys = []
    for i in range(n):
        cluster = clusters[i]
        # Adjust positions to reduce overlap within cluster
        cluster_pos = np.array(cluster_positions[cluster])
        cluster_rad = np.array(cluster_radii_list[cluster])
        centroid = np.mean(cluster_pos, axis=0)
        dist = np.linalg.norm(cluster_pos - centroid, axis=1)
        idx = np.argsort(dist)
        # Move the farthest point to edge of cluster
        max_idx = idx[-1]
        cluster_pos[max_idx] = centroid + (np.random.rand(2) - 0.5) * 0.3
        xs.append(cluster_pos[i][0])
        ys.append(cluster_pos[i][1])
    
    r0 = 0.25 / cols - 1e-3
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
    
    # Non-local reconfiguration: force a controlled expansion of the most isolated cluster
    if res.success:
        v = res.x
        radii = v[2::3]
        # Identify the cluster with the largest separation
        cluster_assignments = np.zeros(n, dtype=int)
        for i in range(n):
            cluster_assignments[i] = clusters[i]
        cluster_separation = []
        for c in range(cols):
            cluster_indices = np.where(cluster_assignments == c)[0]
            if len(cluster_indices) == 0:
                continue
            cluster_positions = v[3*cluster_indices, :2]
            cluster_radii = radii[cluster_indices]
            distances = np.linalg.norm(cluster_positions[:, np.newaxis, :] - cluster_positions[np.newaxis, :, :], axis=-1)
            min_dist = np.min(distances[distances > 0])
            if min_dist > 0:
                cluster_separation.append((c, min_dist))
        if cluster_separation:
            cluster_separation.sort(key=lambda x: -x[1])
            most_isolated_cluster = cluster_separation[0][0]
            cluster_indices = np.where(cluster_assignments == most_isolated_cluster)[0]
            cluster_positions = v[3*cluster_indices, :2]
            cluster_radii = radii[cluster_indices]
            # Expand radii of the most isolated cluster by 5%
            expanded_radii = np.clip(cluster_radii * 1.05, 1e-4, 0.5)
            expanded_positions = cluster_positions.copy()
            # Move cluster positions to optimize spacing
            for i in range(len(cluster_indices)):
                for j in range(len(cluster_indices)):
                    if i != j:
                        dx = expanded_positions[i, 0] - expanded_positions[j, 0]
                        dy = expanded_positions[i, 1] - expanded_positions[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < cluster_radii[i] + cluster_radii[j]:
                            # Move the cluster to a new position
                            direction = np.array([dx, dy]) / dist
                            expanded_positions[i] += direction * (cluster_radii[i] + cluster_radii[j] - dist)
            # Update the decision vector
            perturbed_v = v.copy()
            for i, idx in enumerate(cluster_indices):
                perturbed_v[3*idx] = expanded_positions[i, 0]
                perturbed_v[3*idx+1] = expanded_positions[i, 1]
                perturbed_v[3*idx+2] = expanded_radii[i]
            # Clip radii to ensure they stay within bounds
            perturbed_v[2::3] = np.clip(perturbed_v[2::3], 1e-4, 0.5)
            # Re-evaluate with perturbed parameters
            res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())