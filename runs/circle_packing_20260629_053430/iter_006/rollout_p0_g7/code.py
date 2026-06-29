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
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9, "eps": 1e-8})
    v = res.x if res.success else v0

    # Phase 2: Structural decomposition and independent optimization of sub-components
    # Divide circles into two groups and optimize them separately with different constraints
    group1 = v[0::3][0:13]
    group1_y = v[1::3][0:13]
    group1_r = v[2::3][0:13]
    group2 = v[0::3][13:]
    group2_y = v[1::3][13:]
    group2_r = v[2::3][13:]
    
    # Optimize group1 with different constraints
    v1 = np.concatenate([group1, group1_y, group1_r])
    bounds_group1 = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * 13
    cons_group1 = []
    for i in range(13):
        cons_group1.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_group1.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_group1.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_group1.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(13):
        for j in range(i + 1, 13):
            def constraint_func_group1(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons_group1.append({"type": "ineq", "fun": constraint_func_group1})
    
    res_group1 = minimize(neg_sum_radii, v1, method="SLSQP", bounds=bounds_group1,
                         constraints=cons_group1, options={"maxiter": 500, "ftol": 1e-9, "eps": 1e-8})
    v_group1 = res_group1.x if res_group1.success else v1

    # Optimize group2 with different constraints
    v2 = np.concatenate([group2, group2_y, group2_r])
    bounds_group2 = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * 13
    cons_group2 = []
    for i in range(13):
        cons_group2.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_group2.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_group2.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_group2.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(13):
        for j in range(i + 1, 13):
            def constraint_func_group2(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons_group2.append({"type": "ineq", "fun": constraint_func_group2})
    
    res_group2 = minimize(neg_sum_radii, v2, method="SLSQP", bounds=bounds_group2,
                         constraints=cons_group2, options={"maxiter": 500, "ftol": 1e-9, "eps": 1e-8})
    v_group2 = res_group2.x if res_group2.success else v2

    # Reassemble the configuration with randomized spatial relationships
    v_reassembled = np.concatenate([v_group1[0::3], v_group1[1::3], v_group1[2::3], v_group2[0::3], v_group2[1::3], v_group2[2::3]])
    np.random.seed(42)
    v_reassembled[0::3] += np.random.uniform(-0.05, 0.05, 13)
    v_reassembled[1::3] += np.random.uniform(-0.05, 0.05, 13)
    v_reassembled[2::3] += np.random.uniform(-0.05, 0.05, 13)
    v_reassembled[0::3] += np.random.uniform(-0.05, 0.05, 13)
    v_reassembled[1::3] += np.random.uniform(-0.05, 0.05, 13)
    v_reassembled[2::3] += np.random.uniform(-0.05, 0.05, 13)

    bounds_final = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * 26
    cons_final = []
    for i in range(26):
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_final.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(26):
        for j in range(i + 1, 26):
            def constraint_func_final(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            cons_final.append({"type": "ineq", "fun": constraint_func_final})
    
    res_final = minimize(neg_sum_radii, v_reassembled, method="SLSQP", bounds=bounds_final,
                         constraints=cons_final, options={"maxiter": 500, "ftol": 1e-9, "eps": 1e-8})
    v = res_final.x if res_final.success else v_reassembled

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())