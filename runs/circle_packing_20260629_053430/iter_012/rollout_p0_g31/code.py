import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
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

    # Vectorized overlap constraint with modified distance function
    def vectorized_overlap_constraint(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
        return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)

    # Build constraints
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Add overlap constraints using vectorized function
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dist_sq = (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)
            cons.append({"type": "ineq", "fun": constraint_func})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-9})
    v = res.x if res.success else v0

    # Phase 1: Base optimization
    v = res.x if res.success else v0

    # Phase 2: Topological reconfiguration - decouple position and radius of smallest circle
    # Create new configuration with radius as sole variable, positions constrained to randomized grid
    small_circle_idx = np.argmin(v[2::3])
    small_circle_pos = v[3*small_circle_idx:3*small_circle_idx+3]
    small_circle_radius = v[3*small_circle_idx+2]

    # New configuration: radius as sole variable, positions constrained to randomized grid
    new_v = np.zeros(3 * n)
    new_v[3*small_circle_idx+2] = small_circle_radius
    new_v[3*small_circle_idx:3*small_circle_idx+3] = small_circle_pos

    # Randomized grid for other circles
    np.random.seed(42)
    other_pos = np.random.rand(n-1, 2)
    other_pos = (other_pos - 0.5) * 0.9 + 0.5
    new_v[3*small_circle_idx+3::3] = other_pos[:, 0]
    new_v[3*small_circle_idx+4::3] = other_pos[:, 1]
    new_v[3*small_circle_idx+5::3] = np.full(n-1, 0.01)

    # Bounds for new configuration
    bounds_reconfigured = []
    for i in range(n):
        if i == small_circle_idx:
            bounds_reconfigured += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
        else:
            bounds_reconfigured += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Constraints for new configuration
    cons_reconfigured = []
    for i in range(n):
        if i == small_circle_idx:
            cons_reconfigured.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            cons_reconfigured.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            cons_reconfigured.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            cons_reconfigured.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        else:
            cons_reconfigured.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            cons_reconfigured.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            cons_reconfigured.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            cons_reconfigured.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func_reconfigured(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)
            cons_reconfigured.append({"type": "ineq", "fun": constraint_func_reconfigured})

    res_reconfigured = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds_reconfigured,
                                constraints=cons_reconfigured, options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-9})
    v = res_reconfigured.x if res_reconfigured.success else v

    # Phase 3: Local refinement and final optimization
    res_final = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                         constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-9})
    v = res_final.x if res_final.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())