import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
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

    # Vectorized overlap constraint with modified distance function
    def vectorized_overlap_constraint(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
        return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)

    # Build constraints
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Add overlap constraints using vectorized function
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dist_sq = (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)
            cons.append({"type": "ineq", "fun": constraint_func})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-9})
    v = res.x if res.success else v0

    # Apply radical constraint inversion: prioritize positions over radii
    # and use non-linear scaling to explore new regions
    def scaled_neg_sum_radii(v):
        # Scale radii to force optimizer to consider positions more
        scaled_radii = v[2::3] * 1.1
        return -np.sum(scaled_radii)

    # Rebuild constraints for the new objective
    cons_inverted = []
    for i in range(n):
        cons_inverted.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_inverted.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_inverted.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_inverted.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func_inverted(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)
            cons_inverted.append({"type": "ineq", "fun": constraint_func_inverted})

    res_inverted = minimize(scaled_neg_sum_radii, v, method="SLSQP", bounds=bounds,
                            constraints=cons_inverted, options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-9})
    v = res_inverted.x if res_inverted.success else v

    # Revert to original objective and refine
    res_final = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                         constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-9})
    v = res_final.x if res_final.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())