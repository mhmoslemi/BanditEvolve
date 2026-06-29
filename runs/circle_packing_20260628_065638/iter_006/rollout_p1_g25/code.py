import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a geometric clustering algorithm
    xs = np.random.rand(n)
    ys = np.random.rand(n)
    # Cluster points into groups with spatial proximity
    clusters = np.random.randint(0, 4, size=n)
    for c in range(4):
        mask = clusters == c
        xs[mask] = np.random.uniform(0.2, 0.8, sum(mask))
        ys[mask] = np.random.uniform(0.2, 0.8, sum(mask))
    # Assign cluster-specific offsets to avoid overlap
    for i in range(n):
        xs[i] += 0.1 * np.sin(2 * np.pi * i / n)
        ys[i] += 0.1 * np.cos(2 * np.pi * i / n)
    xs = np.clip(xs, 0.0, 1.0)
    ys = np.clip(ys, 0.0, 1.0)
    
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
        radii = v[2::3]
        # Find cluster centers based on positions
        cluster_centers = np.zeros((4, 2))
        for i in range(n):
            cluster_index = clusters[i]
            cluster_centers[cluster_index] += [v[3*i], v[3*i+1]]
        cluster_centers /= np.sum(np.abs(cluster_centers), axis=1, keepdims=True)
        # Calculate isolation metric for clusters
        cluster_isolation = np.zeros(4)
        for c in range(4):
            dists = np.sum((v[0::3] - cluster_centers[c, 0])**2 + 
                          (v[1::3] - cluster_centers[c, 1])**2, axis=0)
            cluster_isolation[c] = np.mean(dists)
        # Select the most isolated cluster for expansion
        isolated_cluster = np.argmin(cluster_isolation)
        # Apply radius expansion to circles in the isolated cluster
        mask = clusters == isolated_cluster
        perturbed_v = v.copy()
        perturbed_v[3*mask] += 0.02 * np.random.rand(sum(mask))
        perturbed_v[3*mask+1] += 0.02 * np.random.rand(sum(mask))
        perturbed_v[3*mask+2] = np.clip(v[3*mask+2] + 0.02, 1e-4, 0.5)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())