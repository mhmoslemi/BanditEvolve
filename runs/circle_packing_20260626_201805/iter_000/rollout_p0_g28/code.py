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
    x_centers = v0[0::3]
    y_centers = v0[1::3]
    r_centers = v0[2::3]
    x_centers = x_centers.reshape(n, 1)
    y_centers = y_centers.reshape(n, 1)
    r_centers = r_centers.reshape(n, 1)

    dx = x_centers - x_centers.T
    dy = y_centers - y_centers.T
    dist_sq = dx*dx + dy*dy
    radii_sum = r_centers + r_centers.T
    overlap = dist_sq - radii_sum*radii_sum
    overlap = overlap[np.triu_indices(n, 1)]

    # Define constraint function for all pairs
    def vectorized_overlap(v):
        x_centers = v[0::3].reshape(n, 1)
        y_centers = v[1::3].reshape(n, 1)
        r_centers = v[2::3].reshape(n, 1)
        dx = x_centers - x_centers.T
        dy = y_centers - y_centers.T
        dist_sq = dx*dx + dy*dy
        radii_sum = r_centers + r_centers.T
        overlap = dist_sq - radii_sum*radii_sum
        overlap = overlap[np.triu_indices(n, 1)]
        return np.min(overlap)

    # Create constraints for all pairs
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: vectorized_overlap(v)})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())