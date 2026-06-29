import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    xs = np.random.rand(n)
    ys = np.random.rand(n)
    # Group circles into clusters with controlled spacing
    clusters = np.random.choice([1, 2, 3], size=n, p=[0.5, 0.3, 0.2])
    cluster_centers = []
    for c in np.unique(clusters):
        indices = np.where(clusters == c)[0]
        cluster_x = np.random.uniform(0.2, 0.8, size=len(indices))
        cluster_y = np.random.uniform(0.2, 0.8, size=len(indices))
        cluster_centers.extend(zip(cluster_x, cluster_y))
    xs[indices], ys[indices] = zip(*cluster_centers)
    
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
    
    # Non-local reconfiguration: expand the largest cluster to test new topology
    if res.success:
        v = res.x
        clusters = np.random.choice([1, 2, 3], size=n, p=[0.5, 0.3, 0.2])
        cluster_indices = np.arange(n)
        np.random.shuffle(cluster_indices)
        cluster_sizes = np.array([np.sum(clusters == c) for c in np.unique(clusters)])
        largest_cluster = np.argmax(cluster_sizes)
        cluster_mask = clusters == largest_cluster
        # Increase radii of the largest cluster
        v[3*cluster_mask + 2] += 0.005
        # Clip radii to ensure they stay within bounds
        v[2::3] = np.clip(v[2::3], 1e-4, 0.5)
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())