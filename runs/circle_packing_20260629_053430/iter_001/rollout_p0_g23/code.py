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

    def constraint_func_in_bounds(v, i):
        x = v[3*i]
        y = v[3*i+1]
        r = v[3*i+2]
        return np.array([x - r, 1.0 - x - r, y - r, 1.0 - y - r])

    def constraint_func_overlap(v, i, j):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        dist_sq = dx*dx + dy*dy
        r1 = v[3*i+2]
        r2 = v[3*j+2]
        return dist_sq - (r1 + r2)**2

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func_overlap(v, i, j)})

    # Coarse global search
    res_global = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                          constraints=cons, options={"maxiter": 200, "ftol": 1e-8})
    
    # Local refinement using L-BFGS-B
    v_global = res_global.x
    res_local = minimize(neg_sum_radii, v_global, method="L-BFGS-B", bounds=bounds,
                         constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
    
    v = res_local.x if res_local.success else v_global
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())