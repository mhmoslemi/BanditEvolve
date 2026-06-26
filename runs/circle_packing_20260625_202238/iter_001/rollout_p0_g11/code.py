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

    # Vectorize overlap constraints
    x_centers = v0[0::3]
    y_centers = v0[1::3]
    r_centers = v0[2::3]

    # Precompute all pairwise distance constraints
    x_c = x_centers.reshape(n, 1)
    y_c = y_centers.reshape(n, 1)
    r_c = r_centers.reshape(n, 1)

    dx = x_c - x_c.T
    dy = y_c - y_c.T
    dist_sq = dx**2 + dy**2
    r_sum = r_c + r_c.T
    overlap_constraints = dist_sq - r_sum**2

    # Convert to constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Apply constrained reordering mutation
    # Compute constraint tightness for all circles
    # Use initial positions to estimate constraint tightness
    constraint_tightness = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = x_centers[i] - x_centers[j]
                dy = y_centers[i] - y_centers[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r_centers[i] + r_centers[j]:
                    constraint_tightness[i] += 1
                if dist < r_centers[i] + r_centers[j]:
                    constraint_tightness[j] += 1

    # Sort indices based on constraint tightness
    sorted_indices = np.argsort(constraint_tightness)[::-1]

    # Reorder the variables to prioritize tightest constraints
    permuted_v0 = np.zeros_like(v0)
    for i, idx in enumerate(sorted_indices):
        permuted_v0[3*i] = v0[3*idx]
        permuted_v0[3*i+1] = v0[3*idx+1]
        permuted_v0[3*i+2] = v0[3*idx+2]

    # Reorder the constraints to match the new variable order
    permuted_cons = []
    for cons_dict in cons:
        def func(v):
            return cons_dict["fun"](v)
        permuted_cons.append({"type": cons_dict["type"], "fun": func})

    # Optimize with the permuted initial guess and constraints
    res = minimize(neg_sum_radii, permuted_v0, method="SLSQP", bounds=bounds,
                   constraints=permuted_cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else permuted_v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())