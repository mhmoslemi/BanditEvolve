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
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "eps": 1e-8})
    v = res.x if res.success else v0

    # Phase 2: Sub-component optimization with randomized spatial relationships
    sub_cols = 4
    sub_rows = (n + sub_cols - 1) // sub_cols
    sub_indices = []
    for i in range(n):
        sub_row = i // sub_cols
        sub_col = i % sub_cols
        sub_indices.append((sub_row, sub_col))
    
    # Split into sub-components
    sub_components = []
    for r in range(sub_rows):
        for c in range(sub_cols):
            sub_n = 0
            for i in range(n):
                if sub_indices[i][0] == r and sub_indices[i][1] == c:
                    sub_n += 1
            sub_component = []
            for i in range(n):
                if sub_indices[i][0] == r and sub_indices[i][1] == c:
                    sub_component.append(i)
            sub_components.append(sub_component)
    
    # Optimize each sub-component independently
    optimized_v = v.copy()
    for sub in sub_components:
        if len(sub) == 0:
            continue
        sub_v = optimized_v[3*sub[0]:3*sub[0]+3*len(sub)]
        sub_bounds = []
        for _ in range(len(sub)):
            sub_bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
        
        sub_cons = []
        for i in range(len(sub)):
            sub_i = sub[i]
            sub_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            sub_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            sub_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            sub_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                def constraint_func_sub(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                    return dist_sq - min_dist_sq
                sub_cons.append({"type": "ineq", "fun": constraint_func_sub})
        
        res_sub = minimize(neg_sum_radii, sub_v, method="SLSQP", bounds=sub_bounds,
                           constraints=sub_cons, options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-8})
        if res_sub.success:
            optimized_v[3*sub[0]:3*sub[0]+3*len(sub)] = res_sub.x
    
    # Phase 3: Geometric transformation and perturbation
    scale_factor = 1.1
    rotated_v = optimized_v.copy()
    rotated_v[0::3] *= scale_factor
    rotated_v[1::3] *= scale_factor
    rotated_v[1::3] += 0.1  # slight vertical shift to break symmetry
    rotated_v[2::3] *= scale_factor

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

    res_transformed = minimize(neg_sum_radii, rotated_v, method="SLSQP", bounds=bounds_transformed,
                              constraints=cons_transformed, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res_transformed.x if res_transformed.success else optimized_v

    # Phase 4: Local refinement with controlled perturbation
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
                             constraints=cons_perturbed, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res_perturbed.x if res_perturbed.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())