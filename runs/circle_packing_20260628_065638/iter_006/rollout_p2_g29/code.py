import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    # Create clusters with randomized centers and random radius expansion
    centers = np.zeros((n, 2))
    for cluster_id in range(4):  # Divide circles into 4 clusters
        cluster_size = n // 4 + (1 if cluster_id < n % 4 else 0)
        cluster_centers = np.random.rand(cluster_size, 2) * 0.8 + 0.1  # Random cluster center
        cluster_centers += np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]][cluster_id])
        for i in range(cluster_size):
            centers[cluster_id * cluster_size + i] = cluster_centers[i]
    
    # Perturb cluster positions to break symmetry and allow expansion
    for i in range(n):
        centers[i] += np.random.uniform(-0.05, 0.05, 2)
    
    # Initialize radii with a larger initial guess for potential expansion
    r0 = 0.4 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = centers[:, 0]
    v0[1::3] = centers[:, 1]
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

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
    
    # Non-local reconfiguration: expand the most isolated cluster
    if res.success:
        v = res.x
        # Calculate isolation metric: distance to nearest neighbor
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        dist = np.sqrt((x[:, np.newaxis] - x[np.newaxis, :]) ** 2 + (y[:, np.newaxis] - y[np.newaxis, :]) ** 2)
        dist = np.where(dist == 0, np.inf, dist)
        isolation = np.min(dist, axis=1)
        # Select the most isolated cluster (top 5 circles)
        isolated_indices = np.argsort(isolation)[-5:]
        # Apply controlled expansion to these circles
        perturbed_v = v.copy()
        for i in isolated_indices:
            perturbed_v[3*i+2] += 0.05  # Expand radii of isolated circles
            perturbed_v[3*i+2] = np.clip(perturbed_v[3*i+2], 1e-4, 0.5)
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())