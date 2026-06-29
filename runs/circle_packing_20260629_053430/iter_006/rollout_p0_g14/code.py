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

    # Phase 1: Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "eps": 1e-9})
    v = res.x if res.success else v0

    # Phase 2: Sub-component decomposition and optimization
    sub_components = []
    for i in range(0, n, 5):
        sub_component = v[i:i+5]
        sub_components.append(sub_component)
    
    for sub_idx, sub_component in enumerate(sub_components):
        sub_v = np.zeros(3 * 5)
        sub_v[0::3] = sub_component[0::3]
        sub_v[1::3] = sub_component[1::3]
        sub_v[2::3] = sub_component[2::3]
        
        sub_bounds = []
        for _ in range(5):
            sub_bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
        
        sub_cons = []
        for i_sub in range(5):
            sub_cons.append({"type": "ineq", "fun": lambda v, i=i_sub: v[3*i_sub] - v[3*i_sub+2]})
            sub_cons.append({"type": "ineq", "fun": lambda v, i=i_sub: 1.0 - v[3*i_sub] - v[3*i_sub+2]})
            sub_cons.append({"type": "ineq", "fun": lambda v, i=i_sub: v[3*i_sub+1] - v[3*i_sub+2]})
            sub_cons.append({"type": "ineq", "fun": lambda v, i=i_sub: 1.0 - v[3*i_sub+1] - v[3*i_sub+2]})
        for i_sub in range(5):
            for j_sub in range(i_sub + 1, 5):
                def sub_constraint_func(v, i=i_sub, j=j_sub):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                    return dist_sq - min_dist_sq
                sub_cons.append({"type": "ineq", "fun": sub_constraint_func})
        
        res_sub = minimize(neg_sum_radii, sub_v, method="SLSQP", bounds=sub_bounds,
                           constraints=sub_cons, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-9})
        sub_v = res_sub.x if res_sub.success else sub_v
        sub_components[sub_idx] = sub_v
    
    # Reassemble sub-components with randomized spatial relationships
    v = np.zeros(3 * n)
    for i in range(len(sub_components)):
        v[3*i:3*i+3] = sub_components[i][3*i:3*i+3]
        v[3*i+3:3*i+6] = sub_components[i][3*i+3:3*i+6]
        v[3*i+6:3*i+9] = sub_components[i][3*i+6:3*i+9]
    
    # Apply perturbation to break symmetry
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
                             constraints=cons_perturbed, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-9})
    v = res_perturbed.x if res_perturbed.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())