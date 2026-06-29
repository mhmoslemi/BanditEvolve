import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a geometric clustering algorithm
    xs = []
    ys = []
    cluster_centers = np.random.rand(4, 2) * 0.8 + 0.1  # 4 clusters with spacing
    for i in range(n):
        # Assign each circle to the nearest cluster center
        distances = np.sum((np.array([xs, ys])[:, i] - cluster_centers) ** 2, axis=1)
        closest_idx = np.argmin(distances)
        x = cluster_centers[closest_idx, 0] + np.random.uniform(-0.05, 0.05)
        y = cluster_centers[closest_idx, 1] + np.random.uniform(-0.05, 0.05)
        xs.append(x)
        ys.append(y)
    
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

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    # Vectorize the overlap constraints for better performance
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Apply controlled radius expansion to the most isolated cluster
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Compute pairwise distances to identify isolated cluster
        dists = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[0][i] - centers[0][j]
                    dy = centers[1][i] - centers[1][j]
                    dists[i] += np.sqrt(dx*dx + dy*dy)
        isolated_indices = np.argpartition(dists, 2)[:3]  # Top 3 most isolated
        # Expand radii of the most isolated cluster by a controlled amount
        for i in isolated_indices:
            v[3*i+2] = np.clip(v[3*i+2] + 0.005, 1e-4, 0.5)
            # Adjust position to maintain constraint satisfaction
            v[3*i] += np.random.uniform(-0.01, 0.01)
            v[3*i+1] += np.random.uniform(-0.01, 0.01)
        # Re-evaluate with modified parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())