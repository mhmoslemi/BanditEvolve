import numpy as np

def run_packing():
    n = 26
    cols = 5  # Hexagonal grid with 5 columns for better spacing
    rows = (n + cols - 1) // cols  # Ensure enough rows for 26 circles
    
    # Initialize positions using a hexagonal grid pattern
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
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

    # Phase 1: Global optimization with SLSQP focusing on radius expansion
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Phase 2: Local refinement with perturbation and constraint prioritization
    np.random.seed(42)
    perturbation = np.random.uniform(-0.05, 0.05, size=3*n)
    v_perturbed = v + perturbation
    v_perturbed = np.clip(v_perturbed, 0.0, 1.0)
    v_perturbed[2::3] = np.clip(v_perturbed[2::3], 1e-4, 0.5)

    # Refine the perturbed solution with a local optimization emphasizing non-overlapping
    res_perturbed = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                             constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
    v = res_perturbed.x if res_perturbed.success else v

    # Phase 3: Second local refinement with stricter constraints and penalty functions
    def penalty(v):
        penalty = 0
        for i in range(n):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += 1e5 * (abs(x - r) + abs(x + r - 1) + abs(y - r) + abs(y + r - 1))
            for j in range(i + 1, n):
                dx = x - v[3*j]
                dy = y - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (r + v[3*j+2])**2
                if dist_sq < min_dist_sq - 1e-8:
                    penalty += 1e5 * (min_dist_sq - dist_sq)
        return penalty

    # Create a combined objective function with penalty
    def combined_objective(v):
        return neg_sum_radii(v) + penalty(v)

    res_combined = minimize(combined_objective, v, method="SLSQP", bounds=bounds,
                            constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
    v = res_combined.x if res_combined.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())