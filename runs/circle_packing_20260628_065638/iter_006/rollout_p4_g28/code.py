import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    xs = np.random.uniform(0.1, 0.9, n)
    ys = np.random.uniform(0.1, 0.9, n)
    # Group circles into clusters and slightly perturb positions
    cluster_centers = np.random.uniform(0.2, 0.8, (n//2, 2))
    for i in range(n):
        cluster_idx = i // 2
        xs[i] = cluster_centers[cluster_idx, 0] + np.random.uniform(-0.1, 0.1)
        ys[i] = cluster_centers[cluster_idx, 1] + np.random.uniform(-0.1, 0.1)
    
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
    
    # Post-optimization radius expansion for the most isolated cluster
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Calculate isolation scores
        dists = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[0][i] - centers[0][j]
                    dy = centers[1][i] - centers[1][j]
                    dists[i] += np.sqrt(dx*dx + dy*dy)
        isolated_index = np.argmin(dists)
        # Expand radius of the most isolated cluster by 0.003
        cluster_idx = isolated_index // 2
        expansion = 0.003
        v[3*cluster_idx*2 + 2] += expansion
        v[3*cluster_idx*2 + 2] = np.clip(v[3*cluster_idx*2 + 2], 1e-4, 0.5)
        v[3*cluster_idx*2 + 0] += 0.002
        v[3*cluster_idx*2 + 1] += 0.002
        v[3*cluster_idx*2 + 0] = np.clip(v[3*cluster_idx*2 + 0], 0.0, 1.0)
        v[3*cluster_idx*2 + 1] = np.clip(v[3*cluster_idx*2 + 1], 0.0, 1.0)
        # Re-evaluate with modified cluster
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())