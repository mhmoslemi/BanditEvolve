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

    # Calculate constraint tightness for each circle
    constraint_tightness = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            def _constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": _constraint_func})
            constraint_tightness[i] += 1
            constraint_tightness[j] += 1

    # Reorder circles based on constraint tightness
    sorted_indices = np.argsort(constraint_tightness)[::-1]
    reordered_cons = []
    for cons_dict in cons:
        if 'fun' in cons_dict and callable(cons_dict['fun']):
            def _reordered_fun(v, i, j):
                dx = v[3*sorted_indices[i]] - v[3*sorted_indices[j]]
                dy = v[3*sorted_indices[i]+1] - v[3*sorted_indices[j]+1]
                return dx*dx + dy*dy - (v[3*sorted_indices[i]+2] + v[3*sorted_indices[j]+2])**2
            reordered_cons.append({"type": "ineq", "fun": _reordered_fun})
        else:
            reordered_cons.append(cons_dict)

    # Initial optimization with SLSQP
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=reordered_cons, options={"maxiter": 500, "ftol": 1e-9})

    # If optimization fails, use the initial guess
    v = res.x if res.success else v0

    # Local refinement with Nelder-Mead for better convergence
    def local_refinement(v):
        def objective(v):
            return -np.sum(v[2::3])
        res_local = minimize(objective, v, method="Nelder-Mead",
                             bounds=bounds, constraints=reordered_cons,
                             options={"maxiter": 100, "ftol": 1e-9})
        return res_local.x if res_local.success else v

    v = local_refinement(v)

    # Reorder back to original indices
    original_indices = np.argsort(sorted_indices)
    v = v[original_indices*3:(original_indices*3)+3]
    v = np.concatenate([v[0::3], v[1::3], v[2::3]])

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())