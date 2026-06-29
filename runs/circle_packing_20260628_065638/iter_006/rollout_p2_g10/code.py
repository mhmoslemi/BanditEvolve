import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    # Create clusters of circles to allow for more flexible expansion
    clusters = np.random.choice([2, 3], size=n, p=[0.7, 0.3])
    xs = []
    ys = []
    for cluster_idx in range(n):
        # Place cluster centers in a grid with spacing to allow expansion
        cluster_row = cluster_idx // cols
        cluster_col = cluster_idx % cols
        cluster_x = (cluster_col + 0.5) / cols
        cluster_y = (cluster_row + 0.5) / rows
        # Randomly perturb cluster center
        cluster_x += np.random.uniform(-0.05, 0.05)
        cluster_y += np.random.uniform(-0.05, 0.05)
        # Assign cluster members
        for i in range(clusters[cluster_idx]):
            x = cluster_x + np.random.uniform(-0.1, 0.1)
            y = cluster_y + np.random.uniform(-0.1, 0.1)
            xs.append(x)
            ys.append(y)
    
    # Ensure we have exactly 26 circles
    while len(xs) > n:
        xs.pop()
        ys.pop()
    while len(xs) < n:
        xs.append(np.random.uniform(0.1, 0.9))
        ys.append(np.random.uniform(0.1, 0.9))
    
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
    
    # Non-local reconfiguration: isolate the most isolated cluster and expand it
    if res.success:
        v = res.x
        # Compute distance from each circle to its cluster center
        cluster_centers = v[::3], v[1::3]
        cluster_radii = v[2::3]
        cluster_indices = np.random.choice(n, size=int(n * 0.3), replace=False)
        cluster_distances = []
        for i in cluster_indices:
            dx = v[3*i] - cluster_centers[0][i]
            dy = v[3*i+1] - cluster_centers[1][i]
            cluster_distances.append(np.sqrt(dx**2 + dy**2))
        # Find the cluster with the largest distance (most isolated)
        isolated_idx = np.argmax(cluster_distances)
        # Perturb its position and expand its radius
        perturbation = 0.1 * np.random.rand(3)
        perturbed_v = v.copy()
        perturbed_v[3*cluster_indices[isolated_idx]] += perturbation[0]
        perturbed_v[3*cluster_indices[isolated_idx]+1] += perturbation[1]
        perturbed_v[3*cluster_indices[isolated_idx]+2] += perturbation[2]
        # Clip radii to ensure they stay within bounds
        perturbed_v[2::3] = np.clip(perturbed_v[2::3], 1e-4, 0.5)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())