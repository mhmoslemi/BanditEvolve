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

    def create_bounds_and_constraints():
        bounds = []
        cons = []
        for i in range(n):
            bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
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
        return bounds, cons

    bounds, cons = create_bounds_and_constraints()
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Structural decomposition mutation
    subgroups = [
        list(range(0, 8)),  # subgroup 0
        list(range(8, 16)), # subgroup 1
        list(range(16, 24)),# subgroup 2
        list(range(24, 26)) # subgroup 3
    ]
    sub_v = []
    for subgroup in subgroups:
        sub_v.append(v[3*subgroup[0]:3*subgroup[-1]+3])
    
    # Optimize each subgroup independently
    for i, subgroup in enumerate(subgroups):
        sub_bounds = []
        sub_cons = []
        for j in range(len(subgroup)):
            idx = subgroup[j]
            sub_bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
            sub_cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: v[3*i] - v[3*i+2]})
            sub_cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: 1.0 - v[3*i] - v[3*i+2]})
            sub_cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: v[3*i+1] - v[3*i+2]})
            sub_cons.append({"type": "ineq", "fun": lambda v, i=i, j=j: 1.0 - v[3*i+1] - v[3*i+2]})
        for j in range(len(subgroup)):
            for k in range(j + 1, len(subgroup)):
                idx_j = subgroup[j]
                idx_k = subgroup[k]
                def constraint_func(v, i=i, j=j, k=k):
                    dx = v[3*idx_j] - v[3*idx_k]
                    dy = v[3*idx_j+1] - v[3*idx_k+1]
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (v[3*idx_j+2] + v[3*idx_k+2])**2
                    return dist_sq - min_dist_sq
                sub_cons.append({"type": "ineq", "fun": constraint_func})
        sub_res = minimize(neg_sum_radii, sub_v[i], method="SLSQP", bounds=sub_bounds,
                           constraints=sub_cons, options={"maxiter": 300, "ftol": 1e-9})
        sub_v[i] = sub_res.x if sub_res.success else sub_v[i]
    
    # Reassemble optimized subgroups
    for i, subgroup in enumerate(subgroups):
        for j in range(len(subgroup)):
            idx = subgroup[j]
            v[3*idx] = sub_v[i][3*j]
            v[3*idx+1] = sub_v[i][3*j+1]
            v[3*idx+2] = sub_v[i][3*j+2]
    
    # Apply geometric transformation to seed configuration
    scale = 1.1
    rotated_v = v.copy()
    rotated_v[0::3] *= scale
    rotated_v[1::3] *= scale
    rotated_v[1::3] += 0.1  # slight vertical shift to break symmetry

    bounds_transformed, cons_transformed = create_bounds_and_constraints()
    res_transformed = minimize(neg_sum_radii, rotated_v, method="SLSQP", bounds=bounds_transformed,
                              constraints=cons_transformed, options={"maxiter": 500, "ftol": 1e-9})
    v = res_transformed.x if res_transformed.success else v

    # Perform shake heuristic on smallest circles
    indices = np.argsort(v[2::3])
    for i in indices[:5]:  # Shake top 5 smallest circles
        v[3*i] += np.random.uniform(-0.05, 0.05)
        v[3*i+1] += np.random.uniform(-0.05, 0.05)
        v[3*i+2] += np.random.uniform(-0.01, 0.01)

    bounds_shaken, cons_shaken = create_bounds_and_constraints()
    res_shaken = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds_shaken,
                         constraints=cons_shaken, options={"maxiter": 500, "ftol": 1e-9})
    v = res_shaken.x if res_shaken.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())