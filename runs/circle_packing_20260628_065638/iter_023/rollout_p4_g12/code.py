import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints with geometric hashing
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
    
    # Implement shake heuristic to escape local minima by perturbing isolated circles
    for shake_round in range(2):
        if res.success:
            v = res.x
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            
            # Compute pairwise distances
            dists = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dists[i, j] = np.sqrt(dx*dx + dy*dy)
            
            # Find the most isolated circle (least constrained)
            isolation = np.sum(1 / (dists + 1e-8), axis=1)
            isolated_idx = np.argmin(isolation)
            
            # Perturb this circle's position and radius
            perturbation_scale = 0.04 * (1.2 ** shake_round)
            v[3*isolated_idx] += np.random.uniform(-perturbation_scale, perturbation_scale)
            v[3*isolated_idx+1] += np.random.uniform(-perturbation_scale, perturbation_scale)
            v[3*isolated_idx+2] += np.random.uniform(-0.003 * (1.1 ** shake_round), 0.003 * (1.1 ** shake_round))
            
            # Re-optimization
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Final optimization with increased max iterations
    if res.success:
        res = minimize(neg_sum_radii, res.x, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 800, "ftol": 1e-12})
    else:
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())