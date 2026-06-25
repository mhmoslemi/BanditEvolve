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
    for i in range(n):
        def constraint(i):
            def fun(v):
                return v[3*i] - v[3*i + 2]
            return fun
        cons.append({"type": "ineq", "fun": constraint(i)})
        def constraint2(i):
            def fun(v):
                return 1.0 - v[3*i] - v[3*i + 2]
            return fun
        cons.append({"type": "ineq", "fun": constraint2(i)})
        def constraint3(i):
            def fun(v):
                return v[3*i + 1] - v[3*i + 2]
            return fun
        cons.append({"type": "ineq", "fun": constraint3(i)})
        def constraint4(i):
            def fun(v):
                return 1.0 - v[3*i + 1] - v[3*i + 2]
            return fun
        cons.append({"type": "ineq", "fun": constraint4(i)})

    for i in range(n):
        for j in range(i + 1, n):
            def constraint5(i, j):
                def fun(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i + 1] - v[3*j + 1]
                    return dx*dx + dy*dy - (v[3*i + 2] + v[3*j + 2])**2
                return fun
            cons.append({"type": "ineq", "fun": constraint5(i, j)})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Cleanup pass: try to slightly increase radii without moving centers
    def try_inflate_radii(centers, radii):
        n = len(radii)
        max_radius = np.full(n, 0.5)
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            max_radius[i] = min(max_radius[i], 1.0 - r - 1e-6)
            for j in range(n):
                if i != j:
                    dx = x - centers[j][0]
                    dy = y - centers[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    max_radius[i] = min(max_radius[i], (dist - radii[j] - 1e-6))
        delta = np.clip(max_radius - radii, 0, 1e-5)
        new_radii = radii + delta
        return np.column_stack([centers, new_radii]), new_radii

    try_centers, try_radii = try_inflate_radii(centers, radii)
    try_centers = try_centers[:, :2]
    try_radii = try_radii[:, 0]
    
    # Validate the new configuration
    valid, msg = validate_packing(try_centers, try_radii)
    if valid:
        return try_centers, try_radii, float(try_radii.sum())
    else:
        return centers, radii, float(radii.sum())