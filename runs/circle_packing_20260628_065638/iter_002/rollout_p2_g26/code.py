import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a hexagonal grid pattern
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.5 / cols - 1e-3
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
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    
    v = res.x if res.success else v0
    
    # Mutation strategy: replace the initial positions and optimize again
    # with a slightly perturbed initial guess to explore a new region of the search space
    xs_new = np.random.uniform(0.1, 0.9, n)
    ys_new = np.random.uniform(0.1, 0.9, n)
    v_new = np.empty(3 * n)
    v_new[0::3] = xs_new
    v_new[1::3] = ys_new
    v_new[2::3] = np.full(n, r0)
    
    res_mutated = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    
    v = res_mutated.x if res_mutated.success else v
    
    # Local polishing with a penalty function
    def penalty_sum_radii(v, penalty_weight=1e4):
        radii = v[2::3]
        sum_radii = np.sum(radii)
        out_of_bounds = np.sum((v[0::3] - radii) < -1e-12) + np.sum((v[0::3] + radii) > 1 + 1e-12)
        out_of_bounds += np.sum((v[1::3] - radii) < -1e-12) + np.sum((v[1::3] + radii) > 1 + 1e-12)
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                overlap_penalty += max(0, (min_dist_sq - dist_sq) * 1e-6)
        return -sum_radii + penalty_weight * (out_of_bounds + overlap_penalty)

    res_local = minimize(penalty_sum_radii, v, method="L-BFGS-B", bounds=bounds,
                         options={"maxiter": 200, "ftol": 1e-9})
    
    v = res_local.x if res_local.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())