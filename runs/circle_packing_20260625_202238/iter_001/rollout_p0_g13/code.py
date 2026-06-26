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

    # Apply constrained reordering mutation: sort circles based on constraint tightness
    constraint_magnitude = np.abs(overlap_constraints[np.triu_indices(n, 1)])
    constraint_indices = np.argsort(constraint_magnitude)
    
    # Reorder the decision vector based on constraint tightness
    permutation = np.argsort(constraint_indices)
    
    # Reorder the initial guess and constraints
    v0_permuted = np.zeros_like(v0)
    for idx, perm in enumerate(permutation):
        v0_permuted[3*idx] = v0[3*perm]
        v0_permuted[3*idx+1] = v0[3*perm+1]
        v0_permuted[3*idx+2] = v0[3*perm+2]
    
    # Reorder the constraints
    cons_permuted = []
    for idx, perm in enumerate(permutation):
        for jdx, jperm in enumerate(permutation):
            if idx < jdx:
                cons_permuted.append(cons[3*idx + 3*jdx])
    
    res = minimize(neg_sum_radii, v0_permuted, method="SLSQP", bounds=bounds,
                   constraints=cons_permuted, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0_permuted
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())