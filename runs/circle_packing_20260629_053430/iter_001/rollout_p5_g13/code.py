import numpy as np

def run_packing():
    n = 26
    # Use hexagonal grid seeding for better initial packing
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    # Generate hexagonal grid points
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
    # Initial radii: based on average spacing
    r0 = 0.2
    v0 = np.empty(3 * n)
    v0[0::3] = np.array([p[0] for p in points])
    v0[1::3] = np.array([p[1] for p in points])
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # First stage: global optimization with SLSQP for layout
    def constraint_func_global(v, i, j):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
    cons_global = []
    for i in range(n):
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_global.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            cons_global.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func_global(v, i, j)})

    res_global = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                          constraints=cons_global, options={"maxiter": 300, "ftol": 1e-8})
    v_global = res_global.x if res_global.success else v0

    # Second stage: local optimization with L-BFGS-B for fine-tuning
    def constraint_func_local(v, i, j):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
    cons_local = []
    for i in range(n):
        cons_local.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_local.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_local.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_local.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            cons_local.append({"type": "ineq", "fun": lambda v, i=i, j=j: constraint_func_local(v, i, j)})

    res_local = minimize(neg_sum_radii, v_global, method="L-BFGS-B", bounds=bounds,
                         constraints=cons_local, options={"maxiter": 200, "ftol": 1e-9})
    v = res_local.x if res_local.success else v_global
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())