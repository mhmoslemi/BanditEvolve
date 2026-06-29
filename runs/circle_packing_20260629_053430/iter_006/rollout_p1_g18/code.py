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

    # Split into sub-components for independent optimization
    subgroups = [
        [0, 1, 2, 3, 4],  # subgroup 0
        [5, 6, 7, 8, 9],  # subgroup 1
        [10, 11, 12, 13, 14],  # subgroup 2
        [15, 16, 17, 18, 19],  # subgroup 3
        [20, 21, 22, 23, 24],  # subgroup 4
        [25]  # subgroup 5
    ]
    
    # Optimize each subgroup independently with different initial conditions
    for subgroup in subgroups:
        subgroup_v = v.copy()
        for i in subgroup:
            # Introduce random perturbation to break symmetry
            subgroup_v[3*i] += np.random.uniform(-0.05, 0.05)
            subgroup_v[3*i+1] += np.random.uniform(-0.05, 0.05)
            subgroup_v[3*i+2] += np.random.uniform(-0.01, 0.01)
        
        # Rebuild bounds and constraints for subgroup
        bounds_subgroup = []
        for i in subgroup:
            bounds_subgroup += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
        
        cons_subgroup = []
        for i in subgroup:
            cons_subgroup.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            cons_subgroup.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            cons_subgroup.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            cons_subgroup.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        for i in subgroup:
            for j in range(i + 1, n):
                if j in subgroup:
                    def constraint_func_sub(v, i=i, j=j):
                        dx = v[3*i] - v[3*j]
                        dy = v[3*i+1] - v[3*j+1]
                        dist_sq = dx*dx + dy*dy
                        min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                        return dist_sq - min_dist_sq
                    cons_subgroup.append({"type": "ineq", "fun": constraint_func_sub})
        
        res_subgroup = minimize(neg_sum_radii, subgroup_v, method="SLSQP", bounds=bounds_subgroup,
                               constraints=cons_subgroup, options={"maxiter": 500, "ftol": 1e-9})
        v = res_subgroup.x if res_subgroup.success else v

    # Rebuild final configuration
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())