import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    # Better initial layout with hexagonal packing and more refined spacing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Adjusted x and y coordinates to form a hexagonal grid
        x = (col + 0.5) / cols + (row % 2) * (1.0 / (2 * cols))
        y = (row + 0.5) / rows + (row % 2) * (1.0 / (2 * rows))
        xs.append(x)
        ys.append(y)
    rows = (n + cols - 1) // cols
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

    # First stage: global optimization to find good layout
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

    res_global = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                          constraints=cons_global, options={"maxiter": 200, "ftol": 1e-8})

    # Second stage: local optimization with tighter bounds and higher precision
    v = res_global.x if res_global.success else v0
    bounds_local = bounds.copy()
    for i in range(n):
        bounds_local[3*i+2] = (v[3*i+2] * 0.9, v[3*i+2] * 1.1)  # Tighten radius bounds
    cons_local = cons_global.copy()
    res_local = minimize(neg_sum_radii, v, method="L-BFGS-B", bounds=bounds_local,
                         constraints=cons_local, options={"maxiter": 300, "ftol": 1e-10})
    v = res_local.x if res_local.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())