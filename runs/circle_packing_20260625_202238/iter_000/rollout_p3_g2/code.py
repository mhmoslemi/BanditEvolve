import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / cols
    r0 = 0.5 / cols - 1e-3
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
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq + 1e-6
            cons.append({"type": "ineq", "fun": constraint_func})

    # First global optimization with SLSQP
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    
    # Local refinement with penalty function
    def penalty_sum_radii(v):
        sum_r = np.sum(v[2::3])
        # Penalize out-of-bounds
        out_of_bounds = np.sum(np.maximum(0, v[0::3] - v[2::3] - 1e-6) + 
                              np.maximum(0, 1.0 - v[0::3] - v[2::3] - 1e-6) +
                              np.maximum(0, v[1::3] - v[2::3] - 1e-6) + 
                              np.maximum(0, 1.0 - v[1::3] - v[2::3] - 1e-6))
        # Penalize overlaps
        overlap = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                overlap += max(0, min_dist_sq - dist_sq + 1e-6)
        return -sum_r + 1000 * out_of_bounds + 1000 * overlap

    # Local optimization with penalty function
    res_local = minimize(penalty_sum_radii, v, method="L-BFGS-B", bounds=bounds,
                         options={"maxiter": 200, "ftol": 1e-9})
    v = res_local.x
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())