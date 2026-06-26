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
    x_coords = v0[0::3]
    y_coords = v0[1::3]
    r_coords = v0[2::3]

    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Constraint-based reordering mutation
    # Evaluate constraint violations for all circles
    constraint_violations = []
    for i in range(n):
        x, y, r = v[3*i], v[3*i+1], v[3*i+2]
        # Edge constraints
        edge_violation = max(0, (x - r) - 0, (1.0 - x - r), (y - r) - 0, (1.0 - y - r))
        # Overlap constraints
        overlap_violation = 0
        for j in range(n):
            if i == j:
                continue
            dx = x - v[3*j]
            dy = y - v[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r + v[3*j+2]
            if dist < min_dist - 1e-6:
                overlap_violation += max(0, (min_dist - dist))
        constraint_violations.append(edge_violation + overlap_violation)

    # Sort circles by constraint violation (least severe first)
    sorted_indices = np.argsort(constraint_violations)
    # Re-optimize in reverse order of constraint severity
    for idx in sorted_indices[::-1]:
        i = idx
        # Perturb the circle slightly
        v[3*i] += np.random.uniform(-0.02, 0.02)
        v[3*i+1] += np.random.uniform(-0.02, 0.02)
        v[3*i+2] = np.clip(v[3*i+2], 1e-6, 0.5)
    
    # Re-optimize
    res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
    v = res.x if res.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())