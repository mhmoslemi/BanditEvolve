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

    # Vectorize the overlap constraints
    x_centers = v0[0::3]
    y_centers = v0[1::3]
    r_values = v0[2::3]
    num_pairs = n * (n - 1) // 2
    cons_overlap = []

    def constraint_func_overlap(v):
        x_centers = v[0::3]
        y_centers = v[1::3]
        r_values = v[2::3]
        dist_sq = (x_centers[:, np.newaxis] - x_centers[np.newaxis, :]) ** 2 + \
                  (y_centers[:, np.newaxis] - y_centers[np.newaxis, :]) ** 2
        radii_sum = r_values[:, np.newaxis] + r_values[np.newaxis, :]
        return dist_sq - radii_sum ** 2

    cons_overlap.append({"type": "ineq", "fun": constraint_func_overlap})
    
    cons += cons_overlap

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())