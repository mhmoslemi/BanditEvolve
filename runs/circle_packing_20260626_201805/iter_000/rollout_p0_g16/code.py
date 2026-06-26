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

    x_centers = v0[0::3]
    y_centers = v0[1::3]
    r = v0[2::3]

    # Vectorized overlap constraint for all pairs
    x = v0[0::3]
    y = v0[1::3]
    r = v0[2::3]
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    min_dist_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
    overlap = dist_sq - min_dist_sq
    mask = np.triu(np.ones((n, n)), 1).astype(bool)
    overlap[mask] = 0
    cons_overlap = []
    for i in range(n):
        for j in range(i + 1, n):
            cons_overlap.append({"type": "ineq", "fun": lambda v, i=i, j=j: v[3*i+2] + v[3*j+2] - np.sqrt(v[3*i]**2 + v[3*i+1]**2 + v[3*j]**2 + v[3*j+1]**2 - 2 * v[3*i] * v[3*j] - 2 * v[3*i+1] * v[3*j+1])})
    cons += cons_overlap

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())