import numpy as np

def run_packing():
    n = 26
    cols = 5  # Use a fixed hexagonal grid with 5 columns for better spacing
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

    # First phase: global optimization with initial configuration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Second phase: expand smallest circles first
    # Sort circles by current radius
    indices = np.argsort(v[2::3])
    v_sorted = v.copy()
    v_sorted[2::3] = v_sorted[2::3][indices]
    v_sorted = v_sorted.reshape(3, n).T
    v_sorted = v_sorted[indices].flatten()

    # Third phase: local refinement with perturbation and stricter constraints
    perturbation = 0.05
    v_perturbed = v_sorted.copy()
    np.random.seed(42)
    for i in range(n):
        v_perturbed[3*i] += np.random.uniform(-perturbation, perturbation)
        v_perturbed[3*i+1] += np.random.uniform(-perturbation, perturbation)
        v_perturbed[3*i+2] += np.random.uniform(-perturbation, perturbation)

    # Rebuild bounds and constraints for perturbed configuration
    bounds_perturbed = []
    for _ in range(n):
        bounds_perturbed += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    cons_perturbed = []
    for i in range(n):
        cons_perturbed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_perturbed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_perturbed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_perturbed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func_perturbed(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons_perturbed.append({"type": "ineq", "fun": constraint_func_perturbed})

    res_perturbed = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds_perturbed,
                             constraints=cons_perturbed, options={"maxiter": 500, "ftol": 1e-9})
    v = res_perturbed.x if res_perturbed.success else v_sorted

    # Final refinement: expand smallest circles again
    v_final = v.copy()
    for i in range(n):
        # Increase radius of smallest circle if possible
        min_idx = np.argmin(v_final[2::3])
        min_r = v_final[2::3][min_idx]
        # Check if we can increase radius without overlap
        can_increase = True
        for j in range(n):
            if j == min_idx:
                continue
            dx = v_final[3*min_idx] - v_final[3*j]
            dy = v_final[3*min_idx+1] - v_final[3*j+1]
            dist_sq = dx*dx + dy*dy
            min_dist_sq = (min_r + v_final[3*j+2])**2
            if dist_sq < min_dist_sq - 1e-8:
                can_increase = False
                break
        if can_increase:
            # Try to increase radius by a small amount
            max_increase = 0.0
            for j in range(n):
                if j == min_idx:
                    continue
                dx = v_final[3*min_idx] - v_final[3*j]
                dy = v_final[3*min_idx+1] - v_final[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (min_r + v_final[3*j+2])**2
                max_possible_r = (np.sqrt(dist_sq) - v_final[3*j+2])
                max_increase = max(max_increase, max_possible_r - min_r)
            if max_increase > 1e-6:
                v_final[3*min_idx+2] += max_increase * 0.5
                # Ensure the circle stays within the square
                if v_final[3*min_idx] - v_final[3*min_idx+2] < 0:
                    v_final[3*min_idx+2] = v_final[3*min_idx]
                if v_final[3*min_idx] + v_final[3*min_idx+2] > 1:
                    v_final[3*min_idx+2] = 1 - v_final[3*min_idx]
                if v_final[3*min_idx+1] - v_final[3*min_idx+2] < 0:
                    v_final[3*min_idx+2] = v_final[3*min_idx+1]
                if v_final[3*min_idx+1] + v_final[3*min_idx+2] > 1:
                    v_final[3*min_idx+2] = 1 - v_final[3*min_idx+1]

    centers = np.column_stack([v_final[0::3], v_final[1::3]])
    radii = np.clip(v_final[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())