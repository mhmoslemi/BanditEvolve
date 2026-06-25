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
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint})

    # Attempt optimization with tighter tolerances
    try:
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-9, "eps": 1e-9})
    except:
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-9})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Final optimization pass with fixed centers and adjusted radii
    radii_opt = radii.copy()
    if res.success:
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                min_dist = np.sqrt(dx*dx + dy*dy)
                min_radius = (min_dist - 1e-8) / 2
                radii_opt[i] = min(radii_opt[i], min_radius)
                radii_opt[j] = min(radii_opt[j], min_radius)

    # Apply final radii adjustment
    radii = np.clip(radii_opt, 1e-6, None)
    centers = np.column_stack([v[0::3], v[1::3]])

    return centers, radii, float(radii.sum())