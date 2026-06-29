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

    # Structural decomposition mutation: split into sub-components
    subgroups = [list(range(0, 13)), list(range(13, 26))]
    sub_v = []
    for group in subgroups:
        sub_v.append(v[3*group[0]:3*group[-1]+3])
    
    # Optimize each subgroup independently with different constraints
    for i, group in enumerate(subgroups):
        sub_bounds = []
        for _ in range(len(group)):
            sub_bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
        
        sub_cons = []
        for j in range(len(group)):
            sub_cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: v[3*j] - v[3*j+2]})
            sub_cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: 1.0 - v[3*j] - v[3*j+2]})
            sub_cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: v[3*j+1] - v[3*j+2]})
            sub_cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: 1.0 - v[3*j+1] - v[3*j+2]})
        for j in range(len(group)):
            for k in range(j + 1, len(group)):
                def constraint_func_sub(v, i=i, j=j, k=k):
                    dx = v[3*j] - v[3*k]
                    dy = v[3*j+1] - v[3*k+1]
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (v[3*j+2] + v[3*k+2])**2
                    return dist_sq - min_dist_sq
                sub_cons.append({"type": "ineq", "fun": constraint_func_sub})
        
        sub_res = minimize(neg_sum_radii, sub_v[i], method="SLSQP", bounds=sub_bounds,
                           constraints=sub_cons, options={"maxiter": 500, "ftol": 1e-9})
        sub_v[i] = sub_res.x if sub_res.success else sub_v[i]
    
    # Reassemble sub-components with randomized spatial relationships
    v = np.zeros(3 * n)
    for i, group in enumerate(subgroups):
        for j in group:
            v[3*j] = sub_v[i][3*j - 3*len(subgroups[i]) + 3*group.index(j)]
            v[3*j+1] = sub_v[i][3*j - 3*len(subgroups[i]) + 3*group.index(j)+1]
            v[3*j+2] = sub_v[i][3*j - 3*len(subgroups[i]) + 3*group.index(j)+2]
    
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
    
    res_final = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds_final,
                         constraints=cons_final, options={"maxiter": 500, "ftol": 1e-9})
    v = res_final.x if res_final.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())