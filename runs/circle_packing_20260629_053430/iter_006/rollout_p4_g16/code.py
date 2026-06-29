import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
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
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons.append({"type": "ineq", "fun": constraint_func})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Phase 1: Refactor constraint order to prioritize non-overlapping radii
    cons_reordered = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func_reorder(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons_reordered.append({"type": "ineq", "fun": constraint_func_reorder})
    for i in range(n):
        cons_reordered.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_reordered.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_reordered.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_reordered.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    res_reordered = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                             constraints=cons_reordered, options={"maxiter": 500, "ftol": 1e-9})
    v = res_reordered.x if res_reordered.success else v

    # Phase 2: Apply targeted 'shake' heuristic to smallest circles
    # Identify the smallest circles
    radii = v[2::3]
    indices = np.argsort(radii)
    smallest_indices = indices[:5]  # Shake the 5 smallest circles

    # Perturb these circles
    perturbation = 0.05
    v_shaken = v.copy()
    for i in smallest_indices:
        v_shaken[3*i] += np.random.uniform(-perturbation, perturbation)
        v_shaken[3*i+1] += np.random.uniform(-perturbation, perturbation)
        v_shaken[3*i+2] += np.random.uniform(-perturbation, perturbation)

    # Rebuild bounds and constraints for shaken configuration
    bounds_shaken = []
    for _ in range(n):
        bounds_shaken += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    res_shaken = minimize(neg_sum_radii, v_shaken, method="SLSQP", bounds=bounds_shaken,
                          constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res_shaken.x if res_shaken.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())