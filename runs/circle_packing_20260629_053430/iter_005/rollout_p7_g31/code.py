import numpy as np

def run_packing():
    n = 26
    cols = 5  # Fixed hexagonal grid with 5 columns for better spacing
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
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons.append({"type": "ineq", "fun": constraint_func})

    # First phase: global optimization with penalty for out-of-bounds and overlapping circles
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "eps": 1e-8})
    v = res.x if res.success else v0

    # Second phase: apply dual-phase mutation strategy
    # 1. Reorder constraint resolution to prioritize non-overlapping over position
    # 2. Apply controlled perturbation to seed configuration
    def mutation_phase(v):
        # Perturb positions to break symmetry
        perturbation = 0.05
        v_perturbed = v.copy()
        np.random.seed(42)
        for i in range(n):
            v_perturbed[3*i] += np.random.uniform(-perturbation, perturbation)
            v_perturbed[3*i+1] += np.random.uniform(-perturbation, perturbation)
            v_perturbed[3*i+2] += np.random.uniform(-perturbation, perturbation)
        return v_perturbed

    v_mutated = mutation_phase(v)

    # Rebuild bounds and constraints for mutated configuration
    bounds_mutated = []
    for _ in range(n):
        bounds_mutated += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    cons_mutated = []
    for i in range(n):
        cons_mutated.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_mutated.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_mutated.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_mutated.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func_mutated(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons_mutated.append({"type": "ineq", "fun": constraint_func_mutated})

    res_mutated = minimize(neg_sum_radii, v_mutated, method="SLSQP", bounds=bounds_mutated,
                           constraints=cons_mutated, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res_mutated.x if res_mutated.success else v_mutated

    # Third phase: refine with adaptive radius expansion
    radii = v[2::3]
    indices = np.argsort(radii)
    v_refined = v.copy()
    for i in indices[:n//2]:  # Focus on expanding smaller circles
        # Move the circle slightly to give more space for expansion
        v_refined[3*i] += np.random.uniform(-0.01, 0.01)
        v_refined[3*i+1] += np.random.uniform(-0.01, 0.01)
        v_refined[3*i+2] = np.clip(v_refined[3*i+2], 1e-4, 0.5)

    # Final optimization with refined configuration
    bounds_final = []
    for _ in range(n):
        bounds_final += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    cons_final = []
    for i in range(n):
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func_final(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons_final.append({"type": "ineq", "fun": constraint_func_final})

    res_final = minimize(neg_sum_radii, v_refined, method="SLSQP", bounds=bounds_final,
                         constraints=cons_final, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res_final.x if res_final.success else v_refined

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())