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

    # Hybrid optimization: global search with SLSQP and local refinement
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Sort circles by constraint violation severity to prioritize least constrained
    constraint_violations = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dist_sq = dx*dx + dy*dy
            r_sum = v[3*i+2] + v[3*j+2]
            constraint_violations.append(dist_sq - r_sum**2)
    constraint_violations = np.abs(np.array(constraint_violations))
    indices = np.argsort(constraint_violations)
    
    # Reorder circles based on constraint severity
    ordered_centers = np.zeros((n, 2))
    ordered_radii = np.zeros(n)
    for i in range(n):
        idx = indices[i]
        for j in range(n):
            if idx == j:
                ordered_centers[i] = [v[3*j], v[3*j+1]]
                ordered_radii[i] = v[3*j+2]
                break
    
    # Rebuild the decision vector based on reordered circles
    v_reordered = np.zeros(3 * n)
    for i in range(n):
        v_reordered[3*i] = ordered_centers[i, 0]
        v_reordered[3*i+1] = ordered_centers[i, 1]
        v_reordered[3*i+2] = ordered_radii[i]

    # Re-optimize with reordered circles
    res = minimize(neg_sum_radii, v_reordered, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v_reordered

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())