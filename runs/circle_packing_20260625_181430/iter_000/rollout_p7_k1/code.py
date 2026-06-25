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
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    for i in range(n):
        for j in range(i + 1, n):
            def constraint(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i + 1] - v[3*j + 1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i + 2] + v[3*j + 2])**2
                return dist_sq - min_dist_sq
            cons.append({"type": "ineq", "fun": constraint})

    # Initial optimization with SLSQP
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Local refinement with penalty method
    def penalty_func(v):
        penalty = 0.0
        for i in range(n):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            if x - r < 0 or x + r > 1:
                penalty += max(0, 1e-5 - (x - r)) + max(0, 1e-5 - (x + r - 1))
            if y - r < 0 or y + r > 1:
                penalty += max(0, 1e-5 - (y - r)) + max(0, 1e-5 - (y + r - 1))
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                r_i = v[3*i+2]
                r_j = v[3*j+2]
                if dist_sq < (r_i + r_j)**2 - 1e-8:
                    penalty += max(0, 1e-5 - (dist_sq - (r_i + r_j)**2))
        return penalty

    def objective_with_penalty(v):
        return -np.sum(v[2::3]) + 100 * penalty_func(v)

    res2 = minimize(objective_with_penalty, v, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 100, "ftol": 1e-9})
    v = res2.x

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())