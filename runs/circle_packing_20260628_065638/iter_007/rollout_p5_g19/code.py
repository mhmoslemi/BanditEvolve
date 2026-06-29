import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    np.random.seed(42)
    cluster_centers = np.random.rand(cols, rows)
    cluster_centers = (cluster_centers - 0.5) * 0.8 + 0.5  # Center and scale clusters
    cluster_radii = np.full(cols * rows, 0.15)
    cluster_radii[:n] = 0.2  # Increase radii for the first n clusters
    
    # Generate positions by sampling from each cluster
    xs = []
    ys = []
    for i in range(n):
        cluster_idx = i % (cols * rows)
        cluster_col = cluster_idx % cols
        cluster_row = cluster_idx // cols
        x = cluster_centers[cluster_col, cluster_row][0] + np.random.normal(0, 0.05)
        y = cluster_centers[cluster_col, cluster_row][1] + np.random.normal(0, 0.05)
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

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Local perturbation of the densest cluster to test for radius expansion
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Calculate cluster density
        cluster_density = np.zeros(cols * rows)
        for i in range(n):
            cluster_idx = i % (cols * rows)
            cluster_density[cluster_idx] += 1 / (np.pi * radii[i]**2)
        densest_cluster = np.argmax(cluster_density)
        # Perturb the densest cluster's circles
        for i in range(n):
            if i % (cols * rows) == densest_cluster:
                v[3*i + 0] += np.random.uniform(-0.01, 0.01)
                v[3*i + 1] += np.random.uniform(-0.01, 0.01)
                v[3*i + 2] += np.random.uniform(-0.001, 0.001)
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    # Additional refinement for edge cases
    if res.success:
        v = res.x
        radii = v[2::3]
        # Add a small perturbation to circles near boundaries
        for i in range(n):
            x = v[3*i]
            y = v[3*i+1]
            r = v[3*i+2]
            if x < r or x > 1 - r or y < r or y > 1 - r:
                v[3*i + 0] += np.random.uniform(-0.01, 0.01)
                v[3*i + 1] += np.random.uniform(-0.01, 0.01)
                v[3*i + 2] += np.random.uniform(-0.001, 0.001)
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())