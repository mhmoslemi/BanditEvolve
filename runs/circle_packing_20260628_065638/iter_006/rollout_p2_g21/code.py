import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    xs = np.random.rand(n)
    ys = np.random.rand(n)
    # Cluster points in a grid-like pattern
    xs = np.sort(xs)
    ys = np.sort(ys)
    # Create clusters of 5-6 circles with slight spatial separation
    cluster_size = 5
    clusters = np.array_split(np.arange(n), n // cluster_size)
    for i, cluster in enumerate(clusters):
        # Randomly shift cluster to avoid perfect alignment
        shift = np.random.uniform(-0.05, 0.05, size=2)
        xs[cluster] += i * 0.1 + shift[0]
        ys[cluster] += i * 0.1 + shift[1]
    
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

    # Vectorized overlap constraints
    def vectorized_overlap_constraints(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx ** 2 + dy ** 2
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        return dist_sq - r_sum ** 2

    # Convert to list of constraint functions for scipy.optimize
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Convert vectorized constraints to individual constraint functions
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
    
    # Check for isolated cluster and apply controlled radius expansion
    if res.success:
        v = res.x
        # Compute cluster distances
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        # Find clusters with minimal interaction
        cluster_indices = np.arange(n)
        cluster_distances = np.zeros(n)
        for i in range(n):
            min_dist = np.inf
            for j in range(n):
                if i != j:
                    dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    if dist < min_dist:
                        min_dist = dist
            cluster_distances[i] = min_dist
        # Select the cluster with the largest distance
        isolated_index = np.argmax(cluster_distances)
        # Apply controlled radius expansion
        perturbed_v = v.copy()
        perturbed_v[3*isolated_index + 2] += 0.02
        perturbed_v[3*isolated_index + 2] = np.clip(perturbed_v[3*isolated_index + 2], 1e-4, 0.5)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())