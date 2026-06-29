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
            cons.append({"type": "ineq",
                          "fun": lambda v, i=i, j=j: 
                          (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                          - (v[3*i+2] + v[3*j+2])**2})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-11})
    
    # Implement 'shake' heuristic to escape local minima
    if res.success:
        v = res.x
        radii = v[2::3]
        # Perturb smallest circles
        small_radius_indices = np.argsort(radii)[:5]
        for idx in small_radius_indices:
            perturbation = np.random.uniform(-0.02, 0.02, size=2)
            v[3*idx] += perturbation[0]
            v[3*idx+1] += perturbation[1]
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    # Implement multi-step refinement
    if res.success:
        # Step 1: Expand least constrained circle
        v = res.x
        radii = v[2::3]
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmin(min_dists)
        # Expand its radius
        expansion = 0.002
        v[3*least_constrained_idx + 2] += expansion
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    # Step 2: Perturb positions of all circles to find better local optima
    if res.success:
        v = res.x
        random_perturbation = np.random.uniform(-0.02, 0.02, size=3*n)
        v += random_perturbation
        # Clip values to ensure they stay within bounds
        v[0::3] = np.clip(v[0::3], 0.0, 1.0)
        v[1::3] = np.clip(v[1::3], 0.0, 1.0)
        v[2::3] = np.clip(v[2::3], 1e-4, 0.5)
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())