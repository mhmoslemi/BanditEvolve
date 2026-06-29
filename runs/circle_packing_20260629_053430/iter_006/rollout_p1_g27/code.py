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

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Split configuration into sub-components
    sub_n = 13
    sub_v = v[:3*sub_n]
    sub_v2 = v[3*sub_n:]

    # Optimize sub-components with different constraints
    def optimize_sub(sub_v, sub_bounds, sub_cons):
        res = minimize(neg_sum_radii, sub_v, method="SLSQP", bounds=sub_bounds,
                       constraints=sub_cons, options={"maxiter": 500, "ftol": 1e-9})
        return res.x if res.success else sub_v

    # Define bounds and constraints for sub-components
    sub_bounds = []
    for _ in range(sub_n):
        sub_bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    sub_cons = []
    for i in range(sub_n):
        sub_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        sub_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        sub_cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        sub_cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(sub_n):
        for j in range(i + 1, sub_n):
            def constraint_func_sub(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            sub_cons.append({"type": "ineq", "fun": constraint_func_sub})

    sub_v_opt = optimize_sub(sub_v, sub_bounds, sub_cons)

    # Define bounds and constraints for second sub-component
    sub_bounds2 = []
    for _ in range(sub_n):
        sub_bounds2 += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    sub_cons2 = []
    for i in range(sub_n):
        sub_cons2.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        sub_cons2.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        sub_cons2.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        sub_cons2.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(sub_n):
        for j in range(i + 1, sub_n):
            def constraint_func_sub2(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return dist_sq - min_dist_sq
            sub_cons2.append({"type": "ineq", "fun": constraint_func_sub2})

    sub_v2_opt = optimize_sub(sub_v2, sub_bounds2, sub_cons2)

    # Reassemble sub-components with randomized spatial relationships
    v_opt = np.concatenate((sub_v_opt, sub_v2_opt))
    np.random.shuffle(v_opt[3*sub_n:])

    # Final optimization with reassembled configuration
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

    res_final = minimize(neg_sum_radii, v_opt, method="SLSQP", bounds=bounds_final,
                         constraints=cons_final, options={"maxiter": 500, "ftol": 1e-9})
    v = res_final.x if res_final.success else v_opt

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())