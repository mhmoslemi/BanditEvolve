import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a randomized geometric clustering algorithm
    def generate_clustering(n, cols, rows):
        cluster_indices = np.random.choice(n, size=int(n * 0.3), replace=False)
        cluster_x = np.random.uniform(0.1, 0.9, size=len(cluster_indices))
        cluster_y = np.random.uniform(0.1, 0.9, size=len(cluster_indices))
        cluster_r = np.random.uniform(0.05, 0.15, size=len(cluster_indices))
        
        regular_x = []
        regular_y = []
        regular_r = []
        
        for i in range(n):
            if i in cluster_indices:
                continue
            row = i // cols
            col = i % cols
            x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            x += np.random.uniform(-0.05, 0.05)
            y += np.random.uniform(-0.05, 0.05)
            if row % 2 == 1:
                x += 0.5 / cols
            regular_x.append(x)
            regular_y.append(y)
            regular_r.append(0.15 / cols - 1e-3)
        
        xs = np.concatenate([cluster_x, regular_x])
        ys = np.concatenate([cluster_y, np.array(regular_y)])
        rs = np.concatenate([cluster_r, np.array(regular_r)])
        
        # Ensure all coordinates are within [0, 1]
        xs = np.clip(xs, 0.0, 1.0)
        ys = np.clip(ys, 0.0, 1.0)
        return xs, ys, rs
    
    xs, ys, r0 = generate_clustering(n, cols, rows)
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

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
    
    # Post-optimization strategy: expand the most isolated cluster
    if res.success:
        v = res.x
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # Compute distances from each circle to all others
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dists[i, j] = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
        
        # Identify the most isolated cluster (circles with maximum minimum distance)
        min_dists = np.min(dists, axis=1)
        cluster_indices = np.argsort(min_dists)[-int(n * 0.3):]
        
        # Apply controlled expansion to the most isolated cluster
        perturbation = 0.05 * np.random.rand(len(cluster_indices) * 3)
        perturbed_v = v.copy()
        idx = 0
        for i in cluster_indices:
            # Expand radius slightly
            perturbed_v[3*i+2] += 0.01 * np.random.rand()
            # Perturb position slightly
            perturbed_v[3*i] += perturbation[idx]
            perturbed_v[3*i+1] += perturbation[idx+1]
            perturbed_v[3*i+2] += perturbation[idx+2]
            idx += 3
        # Clip radii to ensure they stay within bounds
        perturbed_v[2::3] = np.clip(perturbed_v[2::3], 1e-4, 0.5)
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())