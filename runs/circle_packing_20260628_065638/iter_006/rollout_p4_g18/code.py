import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a hexagonal grid with randomized offsets for perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Introduce random offset to break symmetry and allow better expansion
        x += np.random.uniform(-0.05, 0.05)
        y += np.random.uniform(-0.05, 0.05)
        if row % 2 == 1:
            x += 0.5 / cols
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
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Non-local reconfiguration: replace spatial arrangement with randomized geometric clustering
    if res.success:
        v = res.x
        # Reconfigure positions using a randomized geometric clustering algorithm
        cluster_centers = np.random.rand(4, 2)
        cluster_centers = np.clip(cluster_centers, 0.1, 0.9)
        cluster_radii = np.random.rand(4) * 0.1 + 0.05
        cluster_assignments = np.random.choice(4, n)
        new_x = np.zeros(n)
        new_y = np.zeros(n)
        for i in range(n):
            cluster = cluster_assignments[i]
            new_x[i] = cluster_centers[cluster, 0] + np.random.uniform(-0.1, 0.1)
            new_y[i] = cluster_centers[cluster, 1] + np.random.uniform(-0.1, 0.1)
        
        # Adjust radius based on cluster
        new_r = np.zeros(n)
        for i in range(n):
            cluster = cluster_assignments[i]
            new_r[i] = cluster_radii[cluster] + np.random.uniform(-0.02, 0.02)
        
        # Clip and ensure boundaries
        new_x = np.clip(new_x, 0.0, 1.0)
        new_y = np.clip(new_y, 0.0, 1.0)
        new_r = np.clip(new_r, 1e-4, 0.5)
        
        # Reconstruct decision vector
        perturbed_v = np.zeros(3 * n)
        perturbed_v[0::3] = new_x
        perturbed_v[1::3] = new_y
        perturbed_v[2::3] = new_r
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())