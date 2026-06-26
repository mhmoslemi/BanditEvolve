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

    # Function to compute constraint tightness
    def constraint_tightness(v):
        tightness = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                r_sum = v[3*i+2] + v[3*j+2]
                tight = dist_sq - r_sum*r_sum
                tightness.append(abs(tight))
        # Add boundary constraint tightness
        for i in range(n):
            tightness.append(abs(v[3*i] - v[3*i+2]))
            tightness.append(abs(1.0 - v[3*i] - v[3*i+2]))
            tightness.append(abs(v[3*i+1] - v[3*i+2]))
            tightness.append(abs(1.0 - v[3*i+1] - v[3*i+2]))
        return tightness

    # Initial tightness evaluation
    initial_tightness = constraint_tightness(v0)
    # Create a list of indices and sort by tightness (reverse order for priority)
    indices = np.argsort(initial_tightness)[::-1]
    # Reorder the initial guess based on constraint tightness
    reordered_v0 = np.zeros_like(v0)
    for idx, i in enumerate(indices):
        reordered_v0[3*idx] = v0[3*i]
        reordered_v0[3*idx+1] = v0[3*i+1]
        reordered_v0[3*idx+2] = v0[3*i+2]

    # Rebuild constraints with reordered indices
    new_cons = []
    for i in range(n):
        new_i = np.where(indices == i)[0][0]
        new_cons.append({"type": "ineq", "fun": lambda v, i=new_i: v[3*i] - v[3*i+2]})
        new_cons.append({"type": "ineq", "fun": lambda v, i=new_i: 1.0 - v[3*i] - v[3*i+2]})
        new_cons.append({"type": "ineq", "fun": lambda v, i=new_i: v[3*i+1] - v[3*i+2]})
        new_cons.append({"type": "ineq", "fun": lambda v, i=new_i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            new_i = np.where(indices == i)[0][0]
            new_j = np.where(indices == j)[0][0]
            def constraint_func(v, i=new_i, j=new_j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            new_cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with SLSQP
    res = minimize(neg_sum_radii, reordered_v0, method="SLSQP", bounds=bounds,
                   constraints=new_cons, options={"maxiter": 500, "ftol": 1e-9})

    # If optimization fails, use the reinitialized guess
    v = res.x if res.success else reordered_v0

    # Local refinement with Nelder-Mead for better convergence
    def local_refinement(v):
        def objective(v):
            return -np.sum(v[2::3])
        res_local = minimize(objective, v, method="Nelder-Mead",
                             bounds=bounds, constraints=new_cons,
                             options={"maxiter": 100, "ftol": 1e-9})
        return res_local.x if res_local.success else v

    v = local_refinement(v)

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())