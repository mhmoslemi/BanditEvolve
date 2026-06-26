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

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Constraint-based reordering mutation
    # Evaluate constraint violations for each circle
    constraint_violations = []
    for i in range(n):
        x, y, r = v[3*i], v[3*i+1], v[3*i+2]
        # Boundary violations
        boundary_violation = max(0, (x - r) - 0.0) + max(0, (1.0 - x - r) - 0.0) + \
                           max(0, (y - r) - 0.0) + max(0, (1.0 - y - r) - 0.0)
        # Overlap violations
        overlap_violation = 0
        for j in range(i + 1, n):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            overlap_violation += max(0, (v[3*i+2] + v[3*j+2] - dist) - 1e-8)
        constraint_violations.append(boundary_violation + overlap_violation)

    # Sort circles by constraint violation severity (least constrained first)
    sorted_indices = np.argsort(constraint_violations)
    # Reorder the decision vector and constraints
    reordered_v = np.zeros_like(v)
    reordered_cons = []
    for idx in sorted_indices:
        i = idx
        reordered_v[3*i] = v[3*i]
        reordered_v[3*i+1] = v[3*i+1]
        reordered_v[3*i+2] = v[3*i+2]
        # Update constraints for this circle
        for cons_idx, cons_data in enumerate(cons):
            if cons_data["fun"].__name__ == "lambda v, i=i: v[3*i] - v[3*i+2]":
                reordered_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            elif cons_data["fun"].__name__ == "lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]":
                reordered_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            elif cons_data["fun"].__name__ == "lambda v, i=i: v[3*i+1] - v[3*i+2]":
                reordered_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            elif cons_data["fun"].__name__ == "lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]":
                reordered_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
            elif "constraint_func" in cons_data["fun"].__name__:
                # Find the corresponding i and j
                for j in range(n):
                    for k in range(j + 1, n):
                        if cons_data["fun"].__name__ == f"lambda v, i={j}, j={k}: ...":
                            reordered_cons.append({"type": "ineq", "fun": lambda v, i=j, j=k: \
                                                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2})
                            break
    # Re-optimize with reordered constraints
    res = minimize(neg_sum_radii, reordered_v, method="SLSQP", bounds=bounds,
                   constraints=reordered_cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else reordered_v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())