import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Randomized geometric clustering initialization
    # Generate positions using a clustered random distribution
    xs = []
    ys = []
    cluster_centers = np.random.rand(4, 2) * 0.8 + 0.1  # Four clusters in the square
    for _ in range(n):
        # Randomly select a cluster
        cluster_idx = np.random.choice(4)
        # Generate a point within the cluster's bounds
        x = np.random.rand() * (cluster_centers[cluster_idx, 0] * 0.6) + cluster_centers[cluster_idx, 0] * 0.2
        y = np.random.rand() * (cluster_centers[cluster_idx, 1] * 0.6) + cluster_centers[cluster_idx, 1] * 0.2
        xs.append(x)
        ys.append(y)
    
    r0 = 0.15
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
    
    # Non-local reconfiguration: identify and expand the most isolated cluster
    if res.success:
        v = res.x
        # Calculate isolation scores for each circle
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        isolation_scores = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    isolation_scores[i] += 1.0 / (dist + 1e-6)
        # Find the cluster with the most isolated circles
        cluster_indices = np.argsort(isolation_scores)[-int(n * 0.4):]
        # Increase radii of the most isolated circles
        max_radius = 0.5
        max_radius_increase = max_radius - r[cluster_indices].min()
        r[cluster_indices] += max_radius_increase * np.random.rand(len(cluster_indices))
        r = np.clip(r, 1e-4, max_radius)
        # Create new v with updated radii
        new_v = v.copy()
        new_v[2::3] = r
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())