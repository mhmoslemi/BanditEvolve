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

    # Structural decomposition mutation
    # Split into two sub-components: left (13 circles) and right (13 circles)
    sub1 = v[:3*13]
    sub2 = v[3*13:]

    # Optimize each sub-component independently
    # Sub-component 1: left side
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
                return dist_sq - min_dist_sq
            cons_sub1.append({"type": "ineq", "fun": constraint_func_sub1})

    res_sub1 = minimize(neg_sum_radii, sub1, method="SLSQP", bounds=bounds_sub1,
                        constraints=cons_sub1, options={"maxiter": 500, "ftol": 1e-9})
    sub1 = res_sub1.x if res_sub1.success else sub1

    # Sub-component 2: right side
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
                return dist_sq - min_dist_sq
            cons_sub2.append({"type": "ineq", "fun": constraint_func_sub2})

    res_sub2 = minimize(neg_sum_radii, sub2, method="SLSQP", bounds=bounds_sub2,
                        constraints=cons_sub2, options={"maxiter": 500, "ftol": 1e-9})
    sub2 = res_sub2.x if res_sub2.success else sub2

    # Reassemble with randomized spatial relationships
    v = np.concatenate([sub1, sub2])
    np.random.shuffle(v[3*13:3*26])

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