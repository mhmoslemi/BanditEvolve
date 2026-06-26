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

    # Precompute constraint tightness for reordering
    constraint_tightness = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
            # Estimate constraint tightness based on initial guess
            dx = v0[3*i] - v0[3*j]
            dy = v0[3*i+1] - v0[3*j+1]
            dist_sq = dx*dx + dy*dy
            r_sum_sq = (v0[3*i+2] + v0[3*j+2])**2
            constraint_tightness[i] += abs(dist_sq - r_sum_sq)
            constraint_tightness[j] += abs(dist_sq - r_sum_sq)

    # Reorder circles by constraint tightness
    sorted_indices = np.argsort(constraint_tightness)[::-1]
    reordered_indices = np.argsort(sorted_indices)

    # Reorder the decision vector and constraints based on the new indices
    reordered_v = np.zeros_like(v0)
    for i in range(n):
        reordered_v[3*sorted_indices[i]:3*sorted_indices[i]+3] = v0[3*i:3*i+3]

    reordered_cons = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=sorted_indices[i], j=sorted_indices[j]):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            reordered_cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with SLSQP
    res = minimize(neg_sum_radii, reordered_v, method="SLSQP", bounds=bounds,
                   constraints=reordered_cons, options={"maxiter": 500, "ftol": 1e-9})

    # If optimization fails, use the initial guess
    v = res.x if res.success else reordered_v

    # Local refinement with Nelder-Mead for better convergence
    def local_refinement(v):
        def objective(v):
            return -np.sum(v[2::3])
        res_local = minimize(objective, v, method="Nelder-Mead",
                             bounds=bounds, constraints=reordered_cons,
                             options={"maxiter": 100, "ftol": 1e-9})
        return res_local.x if res_local.success else v

    v = local_refinement(v)

    # Reorder the final result back to original indices
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    original_centers = np.zeros((n, 2))
    original_radii = np.zeros(n)
    for i in range(n):
        original_centers[i] = centers[sorted_indices[i]]
        original_radii[i] = radii[sorted_indices[i]]

    return original_centers, original_radii, float(original_radii.sum())