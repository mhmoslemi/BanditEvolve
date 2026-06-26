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

    # Use a penalty function for constraint violations to allow more flexible optimization
    def constraint_violation_penalty(v):
        penalty = 0.0
        for i in range(n):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            # Boundary constraints
            if x - r < 0:
                penalty += 1e3 * (x - r + 1e-8)
            if x + r > 1:
                penalty += 1e3 * (x + r - 1 + 1e-8)
            if y - r < 0:
                penalty += 1e3 * (y - r + 1e-8)
            if y + r > 1:
                penalty += 1e3 * (y + r - 1 + 1e-8)
            # Overlap constraints
            for j in range(i + 1, n):
                dx = x - v[3*j]
                dy = y - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (r + v[3*j+2])**2
                if dist_sq < min_dist_sq - 1e-8:
                    penalty += 1e3 * (min_dist_sq - dist_sq)
        return penalty

    # Combine objective and penalty
    def total_objective(v):
        return neg_sum_radii(v) + constraint_violation_penalty(v)

    # Define constraints with more robust lambda closures
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

    # Initial optimization with penalty function
    res = minimize(total_objective, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Local refinement with stricter constraints
    if res.success:
        def refine_objective(v):
            return -np.sum(v[2::3])
        
        refined_cons = []
        for i in range(n):
            refined_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            refined_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            refined_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            refined_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        
        for i in range(n):
            for j in range(i + 1, n):
                def refined_constraint_func(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                refined_cons.append({"type": "ineq", "fun": refined_constraint_func})
        
        res_refine = minimize(refine_objective, v, method="SLSQP", bounds=bounds,
                              constraints=refined_cons, options={"maxiter": 200, "ftol": 1e-9})
        v = res_refine.x if res_refine.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())