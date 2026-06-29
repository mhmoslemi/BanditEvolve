import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions using a hexagonal grid pattern
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

    # Define constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Add overlap constraints with penalty function for smoother optimization
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                # Allow small overlap with penalty
                return dist_sq - min_dist_sq + 1e-4 * max(0, (v[3*i+2] + v[3*j+2] - np.sqrt(dist_sq)))
            cons.append({"type": "ineq", "fun": constraint_func})

    # First phase: global optimization with enhanced boundary handling
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "eps": 1e-8})
    v = res.x if res.success else v0
    
    # Geometric transformation: rotate and scale the configuration to create a new layout
    scale = 1.1
    rotation = np.radians(30)
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    rotated_scaled = np.zeros_like(v)
    for i in range(n):
        x = v[3*i]
        y = v[3*i+1]
        r = v[3*i+2]
        # Rotate and scale
        rotated_scaled[3*i] = x * cos_r - y * sin_r
        rotated_scaled[3*i+1] = x * sin_r + y * cos_r
        rotated_scaled[3*i+2] = r
    v_rotated_scaled = rotated_scaled

    # Rebuild bounds and constraints for the transformed configuration
    bounds_rotated = []
    for _ in range(n):
        bounds_rotated += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    cons_rotated = []
    for i in range(n):
        cons_rotated.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_rotated.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_rotated.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_rotated.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func_rotated(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                # Allow small overlap with penalty
                return dist_sq - min_dist_sq + 1e-4 * max(0, (v[3*i+2] + v[3*j+2] - np.sqrt(dist_sq)))
            cons_rotated.append({"type": "ineq", "fun": constraint_func_rotated})

    # Second phase: localized optimization on the transformed configuration
    res_rotated = minimize(neg_sum_radii, v_rotated_scaled, method="SLSQP", bounds=bounds_rotated,
                           constraints=cons_rotated, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res_rotated.x if res_rotated.success else v_rotated_scaled

    # Local refinement with perturbation
    perturbation = 0.05
    v_perturbed = v.copy()
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
                # Allow small overlap with penalty
                return dist_sq - min_dist_sq + 1e-4 * max(0, (v[3*i+2] + v[3*j+2] - np.sqrt(dist_sq)))
            cons_perturbed.append({"type": "ineq", "fun": constraint_func_perturbed})

    res_perturbed = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds_perturbed,
                             constraints=cons_perturbed, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res_perturbed.x if res_perturbed.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())