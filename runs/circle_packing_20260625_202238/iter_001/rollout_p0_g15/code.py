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
    # Sort circles by their constraint tightness
    # We use the initial constraint violation as a proxy for tightness
    initial_violations = np.zeros(n)
    for i in range(n):
        for j in range(i+1, n):
            initial_violations[i] += abs(overlap_constraints[i, j])
            initial_violations[j] += abs(overlap_constraints[i, j])
    sorted_indices = np.argsort(initial_violations)[::-1]  # Descending order

    # Reorder the constraints based on sorted indices
    new_cons = []
    for i in range(n):
        for j in range(i+1, n):
            idx_i = sorted_indices[i]
            idx_j = sorted_indices[j]
            def constraint_func(v, i=idx_i, j=idx_j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            new_cons.append({"type": "ineq", "fun": constraint_func})

    # Reorder the bounds and initial guess
    new_bounds = []
    new_v0 = np.copy(v0)
    for idx in sorted_indices:
        new_bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
        new_v0[3*idx::3] = v0[3*idx::3]
        new_v0[3*idx+1::3] = v0[3*idx+1::3]
        new_v0[3*idx+2::3] = v0[3*idx+2::3]

    # Update constraints
    cons = new_cons

    res = minimize(neg_sum_radii, new_v0, method="SLSQP", bounds=new_bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())