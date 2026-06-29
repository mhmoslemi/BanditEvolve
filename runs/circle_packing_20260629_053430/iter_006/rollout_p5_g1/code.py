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

    # Phase 1: Initial optimization with penalty for out-of-bounds and overlapping circles
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Phase 2: Apply non-linear coordinate mapping transformation
    def transform_coordinates(v):
        # Apply log-polar transformation to the positions and radii
        log_scale = 1.5
        log_rho = np.log(v[2::3] + 1e-6) * log_scale
        log_theta = np.arctan2(v[1::3], v[0::3]) * log_scale
        log_r = v[2::3] * log_scale

        # Transform back to Cartesian coordinates
        new_x = np.cos(log_theta) * np.exp(log_rho)
        new_y = np.sin(log_theta) * np.exp(log_rho)

        # Apply clipping to ensure within the unit square
        new_x = np.clip(new_x, 0.0, 1.0)
        new_y = np.clip(new_y, 0.0, 1.0)
        new_r = np.clip(np.exp(log_r / log_scale) - 1e-6, 1e-4, 0.5)

        # Reconstruct the transformed vector
        transformed_v = np.zeros_like(v)
        transformed_v[0::3] = new_x
        transformed_v[1::3] = new_y
        transformed_v[2::3] = new_r
        return transformed_v

    transformed_v = transform_coordinates(v)
    bounds_transformed = []
    for _ in range(n):
        bounds_transformed += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    cons_transformed = []
    for i in range(n):
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_transformed.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func_transformed(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons_transformed.append({"type": "ineq", "fun": constraint_func_transformed})

    res_transformed = minimize(neg_sum_radii, transformed_v, method="SLSQP", bounds=bounds_transformed,
                              constraints=cons_transformed, options={"maxiter": 500, "ftol": 1e-9})
    v = res_transformed.x if res_transformed.success else transformed_v

    # Phase 3: Local refinement with controlled perturbation
    perturbation = 0.03
    v_perturbed = v.copy()
    np.random.seed(42)
    for i in range(n):
        v_perturbed[3*i] += np.random.uniform(-perturbation, perturbation)
        v_perturbed[3*i+1] += np.random.uniform(-perturbation, perturbation)
        v_perturbed[3*i+2] += np.random.uniform(-perturbation, perturbation)

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
    v = res_perturbed.x if res_perturbed.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())