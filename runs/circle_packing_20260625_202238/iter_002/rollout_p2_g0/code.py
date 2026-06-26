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
    indices = np.arange(n)
    i, j = np.indices((n, n))
    i = np.ravel(i)
    j = np.ravel(j)
    mask = (i < j).ravel()
    valid = np.zeros_like(mask, dtype=bool)
    valid[mask] = True
    i = i[valid]
    j = j[valid]

    def vec_overlap(v):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2

    cons.extend([{"type": "ineq", "fun": lambda v: vec_overlap(v)} for _ in range(len(i))])

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Apply 'shake' heuristic to smallest circles
    if radii.min() > 1e-6:
        small_circle_indices = np.argsort(radii) < 5  # Select smallest 5 circles
        v_shake = v.copy()
        for i in np.where(small_circle_indices)[0]:
            # Perturb center slightly
            v_shake[3*i] += np.random.uniform(-0.01, 0.01)
            v_shake[3*i+1] += np.random.uniform(-0.01, 0.01)
            # Adjust radius to maintain feasibility
            r = v_shake[3*i+2]
            v_shake[3*i+2] = np.clip(r - 0.001, 1e-6, 0.5)

        # Re-optimize with perturbed configuration
        res = minimize(neg_sum_radii, v_shake, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
        v = res.x if res.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())