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
    if res.success:
        v = res.x
    else:
        # Local refinement with tighter bounds if global optimization fails
        v = v0
        for _ in range(10):
            res = minimize(neg_sum_radii, v, method="L-BFGS-B", bounds=bounds,
                           constraints=cons, options={"maxiter": 100, "ftol": 1e-9})
            if res.success:
                v = res.x
                break

    # Constraint-based reordering: sort by constraint severity
    def constraint_severity(v):
        severity = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                r_sum = v[3*i+2] + v[3*j+2]
                severity.append(dist_sq - r_sum**2)
        return severity

    if np.sum(radii) > 0:
        # Sort circles based on constraint severity (least severe first)
        constraint_indices = np.arange(n * (n - 1) // 2)
        severity = np.array(constraint_severity(v))
        sorted_indices = np.argsort(severity)
        # Reorder the decision vector based on sorted constraint indices
        # This is a simplified version for demonstration; full reordering would be more complex
        # For the purpose of this implementation, we use a heuristic to perturb positions

        # Perturb positions of circles with least constraint severity
        perturbation = 0.01
        for idx in range(n):
            # Use a simple heuristic to select circles for perturbation
            if idx < 10:
                v[3*idx] += np.random.uniform(-perturbation, perturbation)
                v[3*idx+1] += np.random.uniform(-perturbation, perturbation)
                v[3*idx+2] = np.clip(v[3*idx+2], 1e-6, 0.5)

        # Re-optimize
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
        v = res.x if res.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())