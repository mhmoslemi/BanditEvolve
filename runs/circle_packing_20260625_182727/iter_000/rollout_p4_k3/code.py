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
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # First stage: Coarse global search with SLSQP
    cons_global = []
    for i in range(n):
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons_global.append({"type": "ineq", "fun": constraint})

    res_global = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                          constraints=cons_global, options={"maxiter": 200, "ftol": 1e-8})
    v_global = res_global.x if res_global.success else v0

    # Second stage: Local refinement with L-BFGS-B
    cons_local = cons_global.copy()
    for i in range(n):
        cons_local.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_local.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_local.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_local.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons_local.append({"type": "ineq", "fun": constraint})

    res_local = minimize(neg_sum_radii, v_global, method="L-BFGS-B", bounds=bounds,
                         constraints=cons_local, options={"maxiter": 300, "ftol": 1e-9})
    v = res_local.x if res_local.success else v_global

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())