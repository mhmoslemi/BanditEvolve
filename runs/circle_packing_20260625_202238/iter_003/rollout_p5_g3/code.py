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

    # Vectorized overlap constraints
    x_centers = v0[0::3]
    y_centers = v0[1::3]
    r_centers = v0[2::3]

    # Precompute all pairwise distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Apply constraint-based reordering mutation
    if np.sum(radii) > 0:
        # Compute constraint violations for all pairs
        constraint_violations = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                violation = max(0, min_dist_sq - dist_sq)
                constraint_violations.append((i, j, violation))
        
        # Sort circles based on the magnitude of constraint violations
        # We will prioritize circles that are most constrained
        constraint_indices = np.argsort([violation for _, _, violation in constraint_violations])
        
        # Reorder the circles in reverse order of constraint severity
        reordered_indices = np.argsort(constraint_indices)
        
        # Reorder the decision vector
        reordered_v = np.zeros_like(v)
        for idx, new_idx in enumerate(reordered_indices):
            reordered_v[3*idx] = v[3*new_idx]
            reordered_v[3*idx+1] = v[3*new_idx+1]
            reordered_v[3*idx+2] = v[3*new_idx+2]
        
        # Re-optimize with reordered variables
        res = minimize(neg_sum_radii, reordered_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
        v = res.x if res.success else reordered_v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())