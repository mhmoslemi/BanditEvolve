import numpy as np

def run_packing():
    n = 26
    # Coarse global search initialization: grid-based layout
    cols = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / cols
    v0_global = np.empty(3 * n)
    v0_global[0::3] = xs
    v0_global[1::3] = ys
    v0_global[2::3] = 0.5 / cols - 1e-3

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Global optimization with SLSQP to find a good layout
    def neg_sum_radii_global(v):
        return -np.sum(v[2::3])

    cons_global = []
    for i in range(n):
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    for i in range(n):
        for j in range(i + 1, n):
            def constraint_global(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i + 1] - v[3*j + 1]
                return dx*dx + dy*dy - (v[3*i + 2] + v[3*j + 2])**2
            cons_global.append({"type": "ineq", "fun": constraint_global})

    res_global = minimize(neg_sum_radii_global, v0_global, method="SLSQP",
                          bounds=bounds, constraints=cons_global,
                          options={"maxiter": 100, "ftol": 1e-8})
    v_global = res_global.x if res_global.success else v0_global

    # Local refinement with L-BFGS-B for higher precision
    v_local = v_global.copy()
    def neg_sum_radii_local(v):
        return -np.sum(v[2::3])

    cons_local = []
    for i in range(n):
        cons_local.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        cons_local.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        cons_local.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        cons_local.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    for i in range(n):
        for j in range(i + 1, n):
            def constraint_local(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i + 1] - v[3*j + 1]
                return dx*dx + dy*dy - (v[3*i + 2] + v[3*j + 2])**2
            cons_local.append({"type": "ineq", "fun": constraint_local})

    res_local = minimize(neg_sum_radii_local, v_local, method="L-BFGS-B",
                         bounds=bounds, constraints=cons_local,
                         options={"maxiter": 200, "ftol": 1e-8})
    v_final = res_local.x if res_local.success else v_local

    centers = np.column_stack([v_final[0::3], v_final[1::3]])
    radii = np.clip(v_final[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())