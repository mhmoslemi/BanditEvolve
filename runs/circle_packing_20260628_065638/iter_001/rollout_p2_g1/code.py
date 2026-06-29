import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    
    # Coarse global search with hexagonal grid initialization
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

    # Coarse global optimization with SLSQP
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

    res_global = minimize(neg_sum_radii, v0_global, method="SLSQP", bounds=bounds,
                          constraints=cons_global, options={"maxiter": 200, "ftol": 1e-8})
    
    # Local refinement with L-BFGS-B
    v_local = res_global.x if res_global.success else v0_global
    v_local = np.clip(v_local, 0, 1)
    
    def neg_sum_radii_local(v):
        return -np.sum(v[2::3])
    
    cons_local = cons_global.copy()
    for i in range(n):
        for j in range(i + 1, n):
            cons_local.append({"type": "ineq",
                               "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})
    
    res_local = minimize(neg_sum_radii_local, v_local, method="L-BFGS-B", bounds=bounds,
                         constraints=cons_local, options={"maxiter": 200, "ftol": 1e-8})
    
    v = res_local.x if res_local.success else v_local
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())