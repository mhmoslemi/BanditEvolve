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

    # Vectorize overlap constraints
    x_centers = v0[0::3]
    y_centers = v0[1::3]
    r_centers = v0[2::3]

    # Precompute all pairwise distance constraints using vectorized operations
    X = v0[0::3].reshape(n, 1)
    Y = v0[1::3].reshape(n, 1)
    R = v0[2::3].reshape(n, 1)

    dx = X - X.T
    dy = Y - Y.T
    dist_sq = dx**2 + dy**2
    r_sum = R + R.T
    overlap_constraints = dist_sq - r_sum**2

    # Convert to constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with SLSQP
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Local refinement using Nelder-Mead to polish the solution
    def local_neg_sum_radii(v):
        return -np.sum(v[2::3])

    res_local = minimize(local_neg_sum_radii, v, method="Nelder-Mead", bounds=bounds,
                         constraints=cons, options={"maxiter": 100, "ftol": 1e-9})
    v = res_local.x if res_local.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())