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
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                r_i = v[3*i+2]
                r_j = v[3*j+2]
                return dx*dx + dy*dy - (r_i + r_j)**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # First phase: geometric distortion
    if radii.shape[0] > 0:
        # Apply a random uniform scaling to the initial configuration
        scale = np.random.uniform(0.9, 1.1)
        v = v * scale
        v[0::3] = np.clip(v[0::3], 0.0, 1.0)
        v[1::3] = np.clip(v[1::3], 0.0, 1.0)
        v[2::3] = np.clip(v[2::3], 1e-4, 0.5)

        # Re-optimize
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
        v = res.x if res.success else v

    # Second phase: localized perturbation of smallest circles
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    if radii.shape[0] > 0:
        sorted_indices = np.argsort(radii)
        smallest_indices = sorted_indices[:5]  # Perturb smallest 5 circles

        # Perturb centers and radii of smallest circles
        for idx in smallest_indices:
            v[3*idx] += np.random.uniform(-0.01, 0.01)
            v[3*idx+1] += np.random.uniform(-0.01, 0.01)
            v[3*idx+2] += np.random.uniform(-0.001, 0.001)

        # Re-optimize with perturbed values
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
        v = res.x if res.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())