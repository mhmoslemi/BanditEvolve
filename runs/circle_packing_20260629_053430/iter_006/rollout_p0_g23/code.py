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

    # Phase 1: Global optimization with initial layout
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "eps": 1e-8})
    v = res.x if res.success else v0
    
    # Phase 2: Apply geometric transformation (scaling and rotation) to seed configuration
    # Scale the configuration to increase spacing
    scale_factor = 1.25
    v_transformed = v.copy()
    v_transformed[0::3] *= scale_factor
    v_transformed[1::3] *= scale_factor
    v_transformed[2::3] *= scale_factor

    # Rebuild bounds and constraints for transformed configuration
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
                # Allow small overlap with penalty
                return dist_sq - min_dist_sq + 1e-4 * max(0, (v[3*i+2] + v[3*j+2] - np.sqrt(dist_sq)))
            cons_transformed.append({"type": "ineq", "fun": constraint_func_transformed})

    res_transformed = minimize(neg_sum_radii, v_transformed, method="SLSQP", bounds=bounds_transformed,
                              constraints=cons_transformed, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res_transformed.x if res_transformed.success else v

    # Phase 3: Structural decomposition and reassembly
    # Split configuration into 2 subcomponents
    sub1 = v[:13]
    sub2 = v[13:]
    
    # Optimize sub1 with modified constraints
    def neg_sum_radii_sub1(v_sub):
        return -np.sum(v_sub[2::3])
    
    bounds_sub1 = []
    for _ in range(13):
        bounds_sub1 += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    cons_sub1 = []
    for i in range(13):
        cons_sub1.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_sub1.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_sub1.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_sub1.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(13):
        for j in range(i + 1, 13):
            def constraint_func_sub1(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                # Allow small overlap with penalty
                return dist_sq - min_dist_sq + 1e-4 * max(0, (v[3*i+2] + v[3*j+2] - np.sqrt(dist_sq)))
            cons_sub1.append({"type": "ineq", "fun": constraint_func_sub1})
    
    res_sub1 = minimize(neg_sum_radii_sub1, sub1, method="SLSQP", bounds=bounds_sub1,
                        constraints=cons_sub1, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    sub1_opt = res_sub1.x if res_sub1.success else sub1
    
    # Optimize sub2 with modified constraints
    def neg_sum_radii_sub2(v_sub):
        return -np.sum(v_sub[2::3])
    
    bounds_sub2 = []
    for _ in range(13):
        bounds_sub2 += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    cons_sub2 = []
    for i in range(13):
        cons_sub2.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_sub2.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_sub2.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_sub2.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(13):
        for j in range(i + 1, 13):
            def constraint_func_sub2(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                # Allow small overlap with penalty
                return dist_sq - min_dist_sq + 1e-4 * max(0, (v[3*i+2] + v[3*j+2] - np.sqrt(dist_sq)))
            cons_sub2.append({"type": "ineq", "fun": constraint_func_sub2})
    
    res_sub2 = minimize(neg_sum_radii_sub2, sub2, method="SLSQP", bounds=bounds_sub2,
                        constraints=cons_sub2, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    sub2_opt = res_sub2.x if res_sub2.success else sub2
    
    # Reassembly with randomized spatial relationships
    v_reassembled = np.concatenate([sub1_opt, sub2_opt])
    # Apply small random perturbation to break symmetry
    np.random.seed(42)
    perturbation = 0.03
    for i in range(n):
        v_reassembled[3*i] += np.random.uniform(-perturbation, perturbation)
        v_reassembled[3*i+1] += np.random.uniform(-perturbation, perturbation)
        v_reassembled[3*i+2] += np.random.uniform(-perturbation, perturbation)

    # Final refinement
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
                # Allow small overlap with penalty
                return dist_sq - min_dist_sq + 1e-4 * max(0, (v[3*i+2] + v[3*j+2] - np.sqrt(dist_sq)))
            cons_final.append({"type": "ineq", "fun": constraint_func_final})

    res_final = minimize(neg_sum_radii, v_reassembled, method="SLSQP", bounds=bounds_final,
                        constraints=cons_final, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res_final.x if res_final.success else v_reassembled

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())