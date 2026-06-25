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
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Attempt optimization with tighter tolerances
    try:
        res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-9, "eps": 1e-8})
        v = res.x if res.success else v0
    except:
        v = v0

    # Final refinement: inflate radii slightly while keeping centers fixed
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    radii = np.minimum(radii + 1e-4, 0.5)  # Ensure radii don't exceed 0.5
    centers = np.column_stack([v[0::3], v[1::3]])

    # Check if refinement caused any constraint violation
    valid, msg = validate_packing(centers, radii)
    if not valid:
        # Fall back to original solution if refinement failed
        centers = np.column_stack([v0[0::3], v0[1::3]])
        radii = np.clip(v0[2::3], 1e-6, None)

    return centers, radii, float(radii.sum())