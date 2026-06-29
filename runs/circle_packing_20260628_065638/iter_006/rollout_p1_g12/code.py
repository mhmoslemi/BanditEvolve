import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    def generate_clustered_positions(n, cols, rows):
        clusters = []
        for i in range(n):
            # Randomly assign to one of the columns
            col = np.random.choice(cols)
            # Randomly assign to a row with some spacing
            row = np.random.choice(rows)
            # Add some random offset within the cluster
            offset_x = np.random.uniform(-0.1, 0.1)
            offset_y = np.random.uniform(-0.1, 0.1)
            x = (col + 0.5 + offset_x) / cols
            y = (row + 0.5 + offset_y) / rows
            clusters.append((x, y))
        return np.array(clusters)
    
    xs = generate_clustered_positions(n, cols, rows)[:, 0]
    ys = generate_clustered_positions(n, cols, rows)[:, 1]
    
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
    
    # Identify the most isolated cluster and apply targeted radius expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Calculate isolation metric (distance to nearest neighbor)
        dists = np.zeros(n)
        for i in range(n):
            min_dist = np.inf
            for j in range(n):
                if i != j:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < min_dist:
                        min_dist = dist
            dists[i] = min_dist
        # Select the cluster with the largest isolation distance
        isolated_index = np.argmax(dists)
        # Apply controlled radius expansion to the isolated cluster
        perturbed_v = v.copy()
        perturbed_v[3*isolated_index + 2] = np.clip(v[3*isolated_index + 2] + 0.02, 1e-4, 0.5)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())