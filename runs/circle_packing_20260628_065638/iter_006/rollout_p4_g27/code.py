import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    xs = []
    ys = []
    # Create initial clusters with random positions and sizes
    cluster_centers = np.random.rand(4, 2) * 0.8 + 0.1
    cluster_radii = np.random.rand(4) * 0.1 + 0.05
    for i in range(n):
        # Assign each circle to a cluster
        cluster_idx = np.random.randint(4)
        x = cluster_centers[cluster_idx, 0] + np.random.normal(0, 0.05)
        y = cluster_centers[cluster_idx, 1] + np.random.normal(0, 0.05)
        # Ensure the circle stays within the unit square
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
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
        # Find the cluster with the most isolated circle
        isolated_indices = np.argsort(dists)
        cluster_indices = np.random.choice(n, size=4, replace=False)
        # Expand the radii of the most isolated cluster
        for i in cluster_indices:
            v[3*i + 2] += 0.02
            # Ensure the radius doesn't exceed the square boundaries
            if v[3*i + 2] > 0.5:
                v[3*i + 2] = 0.5
            # Update the position slightly to maintain spacing
            v[3*i] += np.random.uniform(-0.01, 0.01)
            v[3*i + 1] += np.random.uniform(-0.01, 0.01)
            # Ensure the position stays within bounds
            v[3*i] = np.clip(v[3*i], 0.0, 1.0)
            v[3*i + 1] = np.clip(v[3*i + 1], 0.0, 1.0)
        # Re-optimize with the modified parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())