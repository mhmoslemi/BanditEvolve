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
    # Add boundary constraints
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    # Add overlap constraints with penalty function
    for i in range(n):
        for j in range(i + 1, n):
            def constraint(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i + 1] - v[3*j + 1]
                dist_sq = dx*dx + dy*dy
                r_sum = v[3*i + 2] + v[3*j + 2]
                return dist_sq - r_sum*r_sum
            cons.append({"type": "ineq", "fun": constraint})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Local refinement with tighter bounds
    v_refine = v.copy()
    radii_refine = v[2::3].copy()
    centers_refine = np.column_stack([v[0::3], v[1::3]])

    # Define tight bounds for refinement
    tight_bounds = []
    for i in range(n):
        tight_bounds += [(max(0.0, centers_refine[i, 0] - radii_refine[i]), 
                          min(1.0, centers_refine[i, 0] + radii_refine[i])),
                         (max(0.0, centers_refine[i, 1] - radii_refine[i]), 
                          min(1.0, centers_refine[i, 1] + radii_refine[i])),
                         (radii_refine[i], min(0.5, radii_refine[i] + 0.1))]

    # Refine with tighter bounds and more iterations
    res_refine = minimize(neg_sum_radii, v_refine, method="SLSQP", bounds=tight_bounds,
                          constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})
    v = res_refine.x if res_refine.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())