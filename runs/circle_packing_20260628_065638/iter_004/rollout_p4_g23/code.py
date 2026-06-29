import numpy as np

def run_packing():
    n = 26
    # Use a more randomized initial position to avoid symmetry
    xs = np.random.uniform(0.1, 0.9, n)
    ys = np.random.uniform(0.1, 0.9, n)
    r0 = 0.05  # Larger initial radius to allow more expansion
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = np.full(n, r0)

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
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Global optimization with differential evolution
    res = minimize(neg_sum_radii, v0, method="L-BFGS-B", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})

    # Local refinement with mutation strategy
    if res.success:
        v = res.x
        radii = v[2::3]
        max_radius_index = np.argmax(radii)
        # Perturb the largest circle and re-optimize
        v[3*max_radius_index + 2] += 0.0015  # Small radius increment
        v[3*max_radius_index + 0] += 0.005   # Move circle slightly
        v[3*max_radius_index + 1] += 0.005
        res = minimize(neg_sum_radii, v, method="L-BFGS-B", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())