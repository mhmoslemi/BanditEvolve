import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Randomized geometric clustering initialization
    def random_clustering(n, cols, rows):
        xs = []
        ys = []
        for i in range(n):
            row = i // cols
            col = i % cols
            x = np.random.uniform(0.2, 0.8)
            y = np.random.uniform(0.2, 0.8)
            if row % 2 == 1:
                x += 0.5 / cols
            xs.append(x)
            ys.append(y)
        return xs, ys
    
    xs, ys = random_clustering(n, cols, rows)
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

    # Vectorized overlap constraints
    def vectorized_overlap(v):
        radii = v[2::3]
        x = v[0::3]
        y = v[1::3]
        dist_sq = np.zeros((n, n))
        for i in range(n):
            dx = x - x[i]
            dy = y - y[i]
            dist_sq[:, i] = dx*dx + dy*dy
        return dist_sq - (radii[:, np.newaxis] + radii[np.newaxis, :]) ** 2

    # Create constraints for all pairs using vectorized calculation
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Create constraints for all pairs
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: vectorized_overlap(v)[i, j]})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Controlled radius expansion of most isolated cluster
    if res.success:
        v = res.x
        radii = v[2::3]
        distances = np.zeros(n)
        for i in range(n):
            distances[i] = np.sqrt((v[0::3] - v[0::3][i])**2 + (v[1::3] - v[1::3][i])**2).sum()
        isolated_idx = np.argsort(distances)[-2:]  # Select top 2 most isolated clusters
        
        # Apply controlled radius expansion to the most isolated cluster
        perturbed_v = v.copy()
        for idx in isolated_idx:
            perturbed_v[3*idx+2] += 0.05  # Increase radius by 0.05
            perturbed_v[3*idx+2] = np.clip(perturbed_v[3*idx+2], 1e-4, 0.5)
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())