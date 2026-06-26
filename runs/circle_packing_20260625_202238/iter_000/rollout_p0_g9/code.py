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

    # Vectorized overlap constraints
    x = v0[0::3]
    y = v0[1::3]
    r = v0[2::3]
    X, Y = np.meshgrid(x, y)
    R = np.repeat(r, n, axis=0).reshape(n, n)
    dist_sq = (X - X.T)**2 + (Y - Y.T)**2
    overlap = dist_sq - (R + R.T)**2
    mask = np.triu(np.ones((n, n)), k=1)
    overlap[mask] = np.inf

    def vectorized_overlap(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        dist_sq = (x[:, None] - x[None, :])**2 + (y[:, None] - y[None, :])**2
        min_dist_sq = (r[:, None] + r[None, :])**2
        return np.min(dist_sq - min_dist_sq, axis=1)

    cons.append({"type": "ineq", "fun": vectorized_overlap})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())