import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Randomized geometric clustering initialization
    np.random.seed(42)
    cluster_centers = np.random.rand(3, n)
    cluster_centers[0] = np.clip(cluster_centers[0], 0.1, 0.9)
    cluster_centers[1] = np.clip(cluster_centers[1], 0.1, 0.9)
    cluster_centers[2] = np.clip(cluster_centers[2], 0.01, 0.1)
    
    # Assign each circle to a cluster
    cluster_assignments = np.random.choice(3, n)
    xs = np.zeros(n)
    ys = np.zeros(n)
    for i in range(n):
        cluster = cluster_assignments[i]
        xs[i] = cluster_centers[0, cluster] + np.random.uniform(-0.05, 0.05)
        ys[i] = cluster_centers[1, cluster] + np.random.uniform(-0.05, 0.05)
    
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
    
    # Identify the most isolated cluster
    if res.success:
        v = res.x
        radii = v[2::3]
        # Calculate cluster isolation scores
        cluster_indices = np.bincount(cluster_assignments, minlength=3)
        cluster_positions = v.reshape(n, 3)[:, :2]
        cluster_radii = v.reshape(n, 3)[:, 2]
        cluster_isolation = np.zeros(3)
        for c in range(3):
            cluster_mask = cluster_assignments == c
            cluster_positions_c = cluster_positions[cluster_mask]
            cluster_radii_c = cluster_radii[cluster_mask]
            if cluster_mask.sum() == 0:
                cluster_isolation[c] = np.inf
                continue
            # Calculate distance to other clusters
            distances = np.zeros(cluster_mask.sum())
            for i, pos in enumerate(cluster_positions_c):
                other_positions = cluster_positions[~cluster_mask]
                distances[i] = np.min(np.sqrt(np.sum((pos - other_positions)**2, axis=1)))
            # Calculate minimum distance to other clusters
            cluster_isolation[c] = np.min(distances) - np.sum(cluster_radii_c)
        
        # Select the most isolated cluster
        isolated_cluster = np.argmin(cluster_isolation)
        isolated_indices = np.where(cluster_assignments == isolated_cluster)[0]
        # Increase radii of the isolated cluster by a controlled amount
        max_increase = 0.05
        radii_increase = max_increase * np.random.rand(len(isolated_indices))
        v[3*isolated_indices + 2] += radii_increase
        v[3*isolated_indices + 2] = np.clip(v[3*isolated_indices + 2], 1e-4, 0.5)
        
        # Re-evaluate with modified radii
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())