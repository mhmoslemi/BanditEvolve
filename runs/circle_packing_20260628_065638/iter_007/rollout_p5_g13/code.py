import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using randomized geometric clustering
    np.random.seed(42)
    cluster_centers = np.random.rand(5, 2)
    cluster_centers = np.clip(cluster_centers, 0.1, 0.9)
    
    # Assign each circle to a cluster with some randomness for diversity
    cluster_indices = np.random.randint(0, 5, size=n)
    
    xs = []
    ys = []
    for i in range(n):
        cluster = cluster_indices[i]
        x = cluster_centers[cluster, 0] + np.random.uniform(-0.15, 0.15)
        y = cluster_centers[cluster, 1] + np.random.uniform(-0.15, 0.15)
        # Ensure circle stays within bounds and avoid overlapping clusters
        x = np.clip(x, 0.1, 0.9)
        y = np.clip(y, 0.1, 0.9)
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
    
    # Perturb the most tightly packed cluster to trigger a radical reconfiguration
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        
        # Compute pairwise distances and identify the cluster with the most compact arrangement
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[0][i] - centers[0][j]
                    dy = centers[1][i] - centers[1][j]
                    dists[i, j] = np.sqrt(dx*dx + dy*dy)
        cluster_distances = np.array([np.mean(dists[cluster_indices == c]) for c in range(5)])
        cluster_with_most_packed = np.argmin(cluster_distances)
        
        # Apply controlled expansion to the most tightly packed cluster
        for i in range(n):
            if cluster_indices[i] == cluster_with_most_packed:
                v[3*i + 2] += 0.015  # Increase radius
                v[3*i + 0] += np.random.uniform(-0.01, 0.01)  # Slight position perturbation
                v[3*i + 1] += np.random.uniform(-0.01, 0.01)
        
        # Re-run optimization with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Additional refinement: perturb small circles
    if res.success:
        v = res.x
        radii = v[2::3]
        small_indices = np.where(radii < 0.05)[0]
        for idx in small_indices:
            v[3*idx + 0] += np.random.uniform(-0.01, 0.01)
            v[3*idx + 1] += np.random.uniform(-0.01, 0.01)
            v[3*idx + 2] += np.random.uniform(-0.001, 0.001)
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())