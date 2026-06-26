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

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Hybrid geometric-reinforcement strategy
    if np.sum(radii) > 0:
        # Apply controlled geometric distortion
        distortion = np.random.uniform(-0.02, 0.02, size=3*n)
        v += distortion
        v[2::3] = np.clip(v[2::3], 1e-6, 0.5)
        # Re-optimize with reduced iteration count
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
        v = res.x if res.success else v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)

        # Reinforcement mutation on most recently modified elements
        if np.sum(radii) > 0:
            # Identify circles with largest radii for reinforcement
            sorted_indices = np.argsort(radii)
            large_circle_indices = sorted_indices[-10:]
            # Perturb their positions slightly
            perturbation = 0.005
            for idx in large_circle_indices:
                i = idx
                v[3*i] += np.random.uniform(-perturbation, perturbation)
                v[3*i+1] += np.random.uniform(-perturbation, perturbation)
                v[3*i+2] = np.clip(v[3*i+2], 1e-6, 0.5)
            # Re-optimize
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 100, "ftol": 1e-9})
            v = res.x if res.success else v
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())