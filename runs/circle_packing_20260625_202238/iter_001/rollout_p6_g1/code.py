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

    # Initial optimization with SLSQP
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Shaking heuristic: perturb smallest circles and re-optimize
    def shake(v, perturb=0.05):
        radii = v[2::3]
        indices = np.argsort(radii)
        small_indices = indices[:5]  # Perturb the smallest 5 circles
        v_perturbed = v.copy()
        for idx in small_indices:
            for k in range(3):
                if k == 2:
                    # Perturb radius slightly
                    v_perturbed[3*idx + k] += np.random.uniform(-perturb, perturb)
                else:
                    # Perturb center slightly
                    v_perturbed[3*idx + k] += np.random.uniform(-perturb, perturb)
        return v_perturbed

    v = shake(v)

    # Local refinement using Nelder-Mead for better convergence
    def refiner(v):
        return -np.sum(v[2::3])
    res2 = minimize(refiner, v, method="Nelder-Mead", bounds=bounds,
                    constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
    v = res2.x if res2.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())