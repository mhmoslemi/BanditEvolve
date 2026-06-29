import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    # Generate clusters with randomized positions and radii
    cluster_centers = []
    cluster_radii = []
    for _ in range(5):  # 5 clusters
        # Generate random cluster center
        cx = np.random.uniform(0.1, 0.9)
        cy = np.random.uniform(0.1, 0.9)
        # Generate random radii for circles in the cluster
        cluster_radii.append(np.random.uniform(0.05, 0.15, size=n//5))
        # Generate positions around the cluster center
        for r in cluster_radii[-1]:
            # Place circles around the cluster center with randomized offsets
            theta = np.random.uniform(0, 2*np.pi)
            dx = np.random.uniform(-0.1, 0.1)
            dy = np.random.uniform(-0.1, 0.1)
            x = cx + r * np.cos(theta) + dx
            y = cy + r * np.sin(theta) + dy
            cluster_centers.append((x, y))
    
    # Flatten and convert to numpy arrays
    xs = np.array([c[0] for c in cluster_centers])
    ys = np.array([c[1] for c in cluster_centers])
    r0 = np.array(cluster_radii).flatten()
    
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
    
    # Apply controlled radius expansion to the most isolated cluster
    if res.success:
        v = res.x
        # Identify the most isolated cluster
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        # Compute distances between all pairs
        dists = np.sum((centers[:, np.newaxis] - centers[np.newaxis, :])**2, axis=-1)
        # Find the cluster with the maximum average distance to others
        cluster_indices = np.arange(n)
        cluster_assignments = np.random.choice(5, size=n)
        cluster_distances = []
        for c in range(5):
            cluster_mask = cluster_assignments == c
            cluster_points = centers[cluster_mask]
            avg_dist = np.mean([np.min(dists[cluster_mask, ~cluster_mask]) for _ in range(len(cluster_points))])
            cluster_distances.append(avg_dist)
        most_isolated_idx = np.argmax(cluster_distances)
        # Expand radii of the most isolated cluster
        most_isolated_mask = cluster_assignments == most_isolated_idx
        expanded_radii = radii.copy()
        expanded_radii[most_isolated_mask] *= 1.2
        # Clip radii to ensure they stay within bounds
        expanded_radii = np.clip(expanded_radii, 1e-4, 0.5)
        # Create a new v with expanded radii
        perturbed_v = v.copy()
        perturbed_v[2::3] = expanded_radii
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())