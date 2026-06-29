import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    xs = []
    ys = []
    for i in range(n):
        # Randomly assign cluster centers
        cluster = np.random.randint(0, cols)
        row = np.random.randint(0, rows)
        col = np.random.randint(0, cols)
        x = (col + 0.5 + np.random.uniform(-0.2, 0.2)) / cols
        y = (row + 0.5 + np.random.uniform(-0.2, 0.2)) / rows
        if np.random.random() < 0.3:
            x += np.random.uniform(-0.1, 0.1)
            y += np.random.uniform(-0.1, 0.1)
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
    
    # Non-local reconfiguration: identify and expand the most isolated cluster
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Calculate distance from each circle to all others
        dists = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[0][i] - centers[0][j]
                    dy = centers[1][i] - centers[1][j]
                    dists[i] += np.sqrt(dx*dx + dy*dy)
        # Find the cluster with the largest average distance
        cluster_indices = np.random.choice(n, size=5, replace=False)
        cluster_dists = dists[cluster_indices]
        isolated_index = cluster_indices[np.argmax(cluster_dists)]
        # Expand the radius and slightly perturb the position of the most isolated circle
        v[3*isolated_index + 2] += 0.002
        v[3*isolated_index + 0] += 0.005
        v[3*isolated_index + 1] += 0.005
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())