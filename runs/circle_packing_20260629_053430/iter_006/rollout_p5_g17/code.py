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
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "eps": 1e-8})
    v = res.x if res.success else v0

    # Phase 2: Apply non-linear coordinate transformation for exploration
    def transform_coords(v):
        # Apply log-polar transformation to the x and y coordinates
        # This forces the solver to explore different spatial relationships
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        x_trans = np.log(x + 1e-12)  # Add small value to avoid log(0)
        y_trans = np.log(y + 1e-12)
        r_trans = r * 1.2  # Scale radii to allow more expansion
        return np.concatenate([x_trans, y_trans, r_trans])

    v_transformed = transform_coords(v)
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

    res_transformed = minimize(neg_sum_radii, v_transformed, method="SLSQP", bounds=bounds_transformed,
                              constraints=cons_transformed, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res_transformed.x if res_transformed.success else v

    # Phase 3: Inverse transformation to get back to original coordinates
    v_inverse = v.copy()
    v_inverse[0::3] = np.exp(v_inverse[0::3] - 1e-12)  # Inverse of log(x + 1e-12)
    v_inverse[1::3] = np.exp(v_inverse[1::3] - 1e-12)
    v_inverse[2::3] = v_inverse[2::3] / 1.2

    # Phase 4: Local refinement with controlled perturbation
    perturbation = 0.03
    v_perturbed = v_inverse.copy()
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
                             constraints=cons_perturbed, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res_perturbed.x if res_perturbed.success else v_inverse

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())