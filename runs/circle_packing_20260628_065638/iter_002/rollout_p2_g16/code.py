import numpy as np

def run_packing():
    n = 26
    cols = 5  # Manual adjustment for a hexagonal grid
    rows = (n + cols - 1) // cols  # Ensure enough rows for 26 circles

    # Generate initial positions with a hexagonal grid but with random perturbations
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols + np.random.uniform(-0.05, 0.05)
        y = (row + 0.5) / rows + np.random.uniform(-0.05, 0.05)
        # Offset even rows for hexagonal packing
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)

    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

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
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First optimization with SLSQP
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Targeted radius increment on the largest circle
    radii = v[2::3]
    if np.any(radii > 1e-6):
        largest_circle_idx = np.argmax(radii)
        v[3*largest_circle_idx + 2] = min(radii[largest_circle_idx] + 1e-3, 0.5)

    # Local polishing with a penalty function to refine the solution
    def penalty_sum_radii(v, penalty_weight=1e4):
        radii = v[2::3]
        sum_radii = np.sum(radii)
        # Penalize out-of-bounds
        out_of_bounds = np.sum((v[0::3] - radii) < 0) + np.sum((v[0::3] + radii) > 1)
        out_of_bounds += np.sum((v[1::3] - radii) < 0) + np.sum((v[1::3] + radii) > 1)
        # Penalize overlaps
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                overlap_penalty += max(0, (min_dist_sq - dist_sq) * 1e-6)
        return -sum_radii + penalty_weight * (out_of_bounds + overlap_penalty)

    # Local optimization with penalty function
    res_local = minimize(penalty_sum_radii, v, method="L-BFGS-B", bounds=bounds,
                         options={"maxiter": 200, "ftol": 1e-9})
    
    v = res_local.x if res_local.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())