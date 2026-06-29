import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Generate initial positions with hexagonal packing
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
    
    # Define overlap constraints with penalty for smooth optimization
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                # Penalize small overlaps
                overlap = max(0, (v[3*i+2] + v[3*j+2] - np.sqrt(dist_sq)))
                return dist_sq - min_dist_sq - 1e-4 * overlap
            cons.append({"type": "ineq", "fun": constraint_func})

    # Phase 1: Initial optimization with initial layout
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9, "eps": 1e-8})
    v = res.x if res.success else v0
    
    # Phase 2: Structural decomposition - Split circles into two groups and optimize separately
    group1 = v[0::3][:13]
    group1_y = v[1::3][:13]
    group1_r = v[2::3][:13]
    
    group2 = v[0::3][13:]
    group2_y = v[1::3][13:]
    group2_r = v[2::3][13:]
    
    # Optimize group1 with slight vertical shift to break symmetry
    group1_v = np.empty(3 * 13)
    group1_v[0::3] = group1
    group1_v[1::3] = group1_y + 0.1
    group1_v[2::3] = group1_r
    
    bounds1 = []
    for _ in range(13):
        bounds1 += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    cons1 = []
    for i in range(13):
        cons1.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons1.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons1.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons1.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    for i in range(13):
        for j in range(i + 1, 13):
            def constraint_func1(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                overlap = max(0, (v[3*i+2] + v[3*j+2] - np.sqrt(dist_sq)))
                return dist_sq - min_dist_sq - 1e-4 * overlap
            cons1.append({"type": "ineq", "fun": constraint_func1})

    res1 = minimize(neg_sum_radii, group1_v, method="SLSQP", bounds=bounds1,
                    constraints=cons1, options={"maxiter": 500, "ftol": 1e-9, "eps": 1e-8})
    group1_v = res1.x if res1.success else group1_v
    
    # Optimize group2 with horizontal shift to break symmetry
    group2_v = np.empty(3 * 13)
    group2_v[0::3] = group2 + 0.1
    group2_v[1::3] = group2_y
    group2_v[2::3] = group2_r
    
    bounds2 = []
    for _ in range(13):
        bounds2 += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    cons2 = []
    for i in range(13):
        cons2.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons2.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons2.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons2.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    for i in range(13):
        for j in range(i + 1, 13):
            def constraint_func2(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                overlap = max(0, (v[3*i+2] + v[3*j+2] - np.sqrt(dist_sq)))
                return dist_sq - min_dist_sq - 1e-4 * overlap
            cons2.append({"type": "ineq", "fun": constraint_func2})

    res2 = minimize(neg_sum_radii, group2_v, method="SLSQP", bounds=bounds2,
                    constraints=cons2, options={"maxiter": 500, "ftol": 1e-9, "eps": 1e-8})
    group2_v = res2.x if res2.success else group2_v
    
    # Reconstruct full configuration
    v = np.empty(3 * n)
    v[0::3] = np.concatenate((group1_v[0::3], group2_v[0::3]))
    v[1::3] = np.concatenate((group1_v[1::3], group2_v[1::3]))
    v[2::3] = np.concatenate((group1_v[2::3], group2_v[2::3]))
    
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
                overlap = max(0, (v[3*i+2] + v[3*j+2] - np.sqrt(dist_sq)))
                return dist_sq - min_dist_sq - 1e-4 * overlap
            cons_perturbed.append({"type": "ineq", "fun": constraint_func_perturbed})

    res_perturbed = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds_perturbed,
                             constraints=cons_perturbed, options={"maxiter": 500, "ftol": 1e-9, "eps": 1e-8})
    v = res_perturbed.x if res_perturbed.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())