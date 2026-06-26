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

    # Calculate constraint tightness for reordering
    constraint_tightness = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
            # Estimate tightness based on initial guess
            dx = v0[3*i] - v0[3*j]
            dy = v0[3*i+1] - v0[3*j+1]
            dist_sq = dx*dx + dy*dy
            r_sum = v0[3*i+2] + v0[3*j+2]
            tightness = dist_sq - r_sum*r_sum
            constraint_tightness[i] += abs(tightness)
            constraint_tightness[j] += abs(tightness)

    # Sort indices based on constraint tightness
    sorted_indices = np.argsort(constraint_tightness)[::-1]
    # Reorder the decision vector and constraints
    reordered_v = np.zeros(3 * n)
    reordered_cons = []
    for i in range(n):
        idx = sorted_indices[i]
        reordered_v[3*i] = v0[3*idx]
        reordered_v[3*i+1] = v0[3*idx+1]
        reordered_v[3*i+2] = v0[3*idx+2]
        # Reorder constraints
        for j in range(n):
            if j == idx:
                continue
            # Find and reorder constraint functions
            for k, con in enumerate(cons):
                if con["fun"].__code__.co_freevars == ('i', 'j') and con["fun"].__defaults__ == (i, j):
                    new_i = i
                    new_j = j
                    def new_constraint_func(v, i=new_i, j=new_j):
                        dx = v[3*i] - v[3*j]
                        dy = v[3*i+1] - v[3*j+1]
                        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                    reordered_cons.append({"type": "ineq", "fun": new_constraint_func})
        for k, con in enumerate(cons):
            if con["fun"].__code__.co_freevars == ('i', 'j') and con["fun"].__defaults__ == (idx, i):
                new_i = i
                new_j = idx
                def new_constraint_func(v, i=new_i, j=new_j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                reordered_cons.append({"type": "ineq", "fun": new_constraint_func})

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

    # Reorder back to original index order
    original_v = np.zeros(3 * n)
    for i in range(n):
        idx = sorted_indices[i]
        original_v[3*idx] = v[3*i]
        original_v[3*idx+1] = v[3*i+1]
        original_v[3*idx+2] = v[3*i+2]

    centers = np.column_stack([original_v[0::3], original_v[1::3]])
    radii = np.clip(original_v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())