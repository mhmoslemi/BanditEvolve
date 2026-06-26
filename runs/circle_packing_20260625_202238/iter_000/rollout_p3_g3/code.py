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
    
    # Add a penalty function to handle overlapping and out-of-bounds constraints
    def penalty(v):
        penalty = 0.0
        # Check for out-of-bounds
        for i in range(n):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += 1000.0 * (max(0, x - r) + max(0, 1 - x - r) + max(0, y - r) + max(0, 1 - y - r))
        # Check for overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                r_sum = v[3*i+2] + v[3*j+2]
                if dist_sq < r_sum*r_sum - 1e-12:
                    penalty += 10000.0 * (r_sum*r_sum - dist_sq)
        return penalty

    # Modify the constraints to include penalty terms
    for i in range(n):
        for j in range(i+1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Use a hybrid approach with a global optimization followed by a local refinement
    # First, use a global optimization with a modified objective function that includes penalty
    def objective_with_penalty(v):
        return neg_sum_radii(v) + penalty(v)

    res = minimize(objective_with_penalty, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    # Perform a local refinement with the original objective
    res_refine = minimize(neg_sum_radii, v, method="L-BFGS-B", bounds=bounds,
                          constraints=cons, options={"maxiter": 100, "ftol": 1e-9})
    v = res_refine.x if res_refine.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())