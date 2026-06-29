import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    # Hexagonal grid seeding with offset for better packing
    centers = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols + (row % 2) * (1.0 / (2 * cols))
        y = row / rows + (row % 2) * (1.0 / (2 * rows))
        centers.append([x, y])
    # Initial radii based on hexagonal packing
    r0 = np.full(n, 0.05)
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(centers)[:, 0]
    v0[1::3] = np.array(centers)[:, 1]
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Coarse global search with SLSQP
    def coarse_search(v0):
        cons = []
        for i in range(n):
            cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        for i in range(n):
            for j in range(i + 1, n):
                def constraint_func(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                cons.append({"type": "ineq", "fun": constraint_func})
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-8})
        return res.x if res.success else v0

    # Local refinement with L-BFGS-B
    def local_refinement(v0):
        cons = []
        for i in range(n):
            cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        for i in range(n):
            for j in range(i + 1, n):
                def constraint_func(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                cons.append({"type": "ineq", "fun": constraint_func})
        res = minimize(neg_sum_radii, v0, method="L-BFGS-B", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
        return res.x if res.success else v0

    v = coarse_search(v0)
    v = local_refinement(v)
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())