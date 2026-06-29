import numpy as np

def run_packing():
    n = 26
    # Two-stage optimization pipeline: coarse global search followed by fine-tuned local optimization
    
    # First stage: global search with SLSQP for layout
    cols = int(np.ceil(np.sqrt(n)))
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
    v0_global = np.empty(3 * n)
    v0_global[0::3] = np.array(xs)
    v0_global[1::3] = np.array(ys)
    v0_global[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    cons_global = []
    for i in range(n):
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons_global.append({"type": "ineq", "fun": constraint_func})

    # First optimization stage with SLSQP
    res_global = minimize(neg_sum_radii, v0_global, method="SLSQP", bounds=bounds,
                         constraints=cons_global, options={"maxiter": 300, "ftol": 1e-8})
    v_global = res_global.x if res_global.success else v0_global

    # Second stage: local optimization with L-BFGS-B for fine-tuning radii
    v_local = v_global.copy()
    cons_local = cons_global.copy()

    # Remove constraints on center positions to allow more flexibility in local optimization
    # Only keep constraints on radii bounds and circle-circle separation
    cons_local = [con for con in cons_local if con["type"] == "ineq" and "fun" in con]

    res_local = minimize(neg_sum_radii, v_local, method="L-BFGS-B", bounds=bounds,
                         constraints=cons_local, options={"maxiter": 200, "ftol": 1e-9})
    v_final = res_local.x if res_local.success else v_global

    # Extract final positions and radii
    centers = np.column_stack([v_final[0::3], v_final[1::3]])
    radii = np.clip(v_final[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())