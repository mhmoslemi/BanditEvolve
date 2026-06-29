import numpy as np

def run_packing():
    n = 26
    # Stage 1: Coarse global search with hexagonal seeding
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    points = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5 * (i % 2)) / cols
            y = i / rows
            points.append((x, y))
            if len(points) == n:
                break
        if len(points) == n:
            break
    r0 = 0.25
    v0_global = np.empty(3 * n)
    v0_global[0::3] = np.array([p[0] for p in points])
    v0_global[1::3] = np.array([p[1] for p in points])
    v0_global[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii_global(v):
        return -np.sum(v[2::3])

    cons_global = []
    for i in range(n):
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func_global(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons_global.append({"type": "ineq", "fun": constraint_func_global})

    res_global = minimize(neg_sum_radii_global, v0_global, method="SLSQP", bounds=bounds,
                          constraints=cons_global, options={"maxiter": 200, "ftol": 1e-8})
    v_global = res_global.x if res_global.success else v0_global

    # Stage 2: Local refinement using L-BFGS-B
    v_local = np.copy(v_global)
    radii_local = v_local[2::3]
    centers_local = np.column_stack([v_local[0::3], v_local[1::3]])

    def neg_sum_radii_local(v):
        return -np.sum(v[2::3])

    cons_local = []
    for i in range(n):
        cons_local.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_local.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_local.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_local.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func_local(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons_local.append({"type": "ineq", "fun": constraint_func_local})

    res_local = minimize(neg_sum_radii_local, v_local, method="L-BFGS-B", bounds=bounds,
                         constraints=cons_local, options={"maxiter": 300, "ftol": 1e-9})
    v = res_local.x if res_local.success else v_global
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())