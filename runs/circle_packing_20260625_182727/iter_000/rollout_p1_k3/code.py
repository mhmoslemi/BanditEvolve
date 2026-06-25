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

    # Final cleanup pass: try to inflate radii slightly without moving centers
    # Check if the current packing is valid
    valid, msg = validate_packing(centers, radii)
    if valid:
        # Try to increase radii slightly while maintaining center positions
        new_radii = radii * 1.001
        new_radii = np.clip(new_radii, 1e-6, 0.5)
        # Check if new_radii is valid
        valid_new, msg_new = validate_packing(centers, new_radii)
        if valid_new:
            radii = new_radii
    return centers, radii, float(radii.sum())