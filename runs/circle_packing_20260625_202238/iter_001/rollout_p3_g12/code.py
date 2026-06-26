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

    # Calculate constraint tightness and reorder circles for mutation
    constraint_tightness = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
            # Estimate tightness based on initial position
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            r_sum = r0[i] + r0[j]
            constraint_tightness[i] += dist_sq - r_sum**2
            constraint_tightness[j] += dist_sq - r_sum**2

    # Reorder circles based on constraint tightness
    sorted_indices = np.argsort(constraint_tightness)[::-1]
    reordered_indices = np.array([sorted_indices[i] for i in range(n)])
    reorder_map = np.zeros(n, dtype=int)
    for i in range(n):
        reorder_map[sorted_indices[i]] = i

    # Reorder the decision vector and constraints
    v_reorder = np.zeros_like(v0)
    for i in range(n):
        v_reorder[3*i] = v0[3*reorder_map[i]]
        v_reorder[3*i+1] = v0[3*reorder_map[i]+1]
        v_reorder[3*i+2] = v0[3*reorder_map[i]+2]

    # Reorder constraints
    new_cons = []
    for i in range(n):
        new_cons.append({"type": "ineq", "fun": lambda v, i=reorder_map[i]: v[3*i] - v[3*i+2]})
        new_cons.append({"type": "ineq", "fun": lambda v, i=reorder_map[i]: 1.0 - v[3*i] - v[3*i+2]})
        new_cons.append({"type": "ineq", "fun": lambda v, i=reorder_map[i]: v[3*i+1] - v[3*i+2]})
        new_cons.append({"type": "ineq", "fun": lambda v, i=reorder_map[i]: 1.0 - v[3*i+1] - v[3*i+2]})

    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=reorder_map[i], j=reorder_map[j]):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            new_cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with SLSQP
    res = minimize(neg_sum_radii, v_reorder, method="SLSQP", bounds=bounds,
                   constraints=new_cons, options={"maxiter": 500, "ftol": 1e-9})

    # If optimization fails, use the initial guess
    v = res.x if res.success else v_reorder

    # Local refinement with Nelder-Mead for better convergence
    def local_refinement(v):
        def objective(v):
            return -np.sum(v[2::3])
        res_local = minimize(objective, v, method="Nelder-Mead",
                             bounds=bounds, constraints=new_cons,
                             options={"maxiter": 100, "ftol": 1e-9})
        return res_local.x if res_local.success else v

    v = local_refinement(v)

    # Reorder back to original indices
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())