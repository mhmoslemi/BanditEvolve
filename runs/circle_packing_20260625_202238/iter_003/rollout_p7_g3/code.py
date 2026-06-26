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

    def vectorized_overlap_constraint(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        dist_sq = (x[:, np.newaxis] - x[np.newaxis, :]) ** 2 + (y[:, np.newaxis] - y[np.newaxis, :]) ** 2
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        return dist_sq - r_sum ** 2

    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    if np.sum(radii) > 0:
        # Reorder circles by constraint tightness
        def constraint_tightness(v):
            x = v[0::3]
            y = v[1::3]
            r = v[2::3]
            dist_sq = (x[:, np.newaxis] - x[np.newaxis, :]) ** 2 + (y[:, np.newaxis] - y[np.newaxis, :]) ** 2
            r_sum = r[:, np.newaxis] + r[np.newaxis, :]
            return np.abs(dist_sq - r_sum ** 2)

        tightness = constraint_tightness(v)
        indices = np.argsort(tightness, axis=None)
        permuted_v = np.zeros_like(v)
        permuted_v[::3] = v[indices[::3]]
        permuted_v[1::3] = v[indices[1::3]]
        permuted_v[2::3] = v[indices[2::3]]
        v = permuted_v

        # Re-optimize after reordering
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
        v = res.x if res.success else v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)

    return centers, radii, float(radii.sum())