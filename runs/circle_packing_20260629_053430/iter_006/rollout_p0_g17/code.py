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

    # Phase 2: Structural decomposition and reassembly
    # Split into sub-components: top half and bottom half
    sub1 = v[0:13]
    sub2 = v[13:26]
    
    # Optimize each sub-component with different constraints
    def optimize_sub(v_sub):
        bounds_sub = []
        for _ in range(len(v_sub)//3):
            bounds_sub += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
        
        def neg_sum_radii_sub(v_sub):
            return -np.sum(v_sub[2::3])
        
        cons_sub = []
        for i in range(len(v_sub)//3):
            cons_sub.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
            cons_sub.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            cons_sub.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
            cons_sub.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        for i in range(len(v_sub)//3):
            for j in range(i + 1, len(v_sub)//3):
                def constraint_func_sub(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                    return dist_sq - min_dist_sq
                cons_sub.append({"type": "ineq", "fun": constraint_func_sub})
        
        res_sub = minimize(neg_sum_radii_sub, v_sub, method="SLSQP", bounds=bounds_sub,
                           constraints=cons_sub, options={"maxiter": 500, "ftol": 1e-9})
        return res_sub.x if res_sub.success else v_sub
    
    sub1_opt = optimize_sub(sub1)
    sub2_opt = optimize_sub(sub2)
    
    # Reassemble with randomized spatial relationships
    v_reassembled = np.concatenate([sub1_opt, sub2_opt])
    v_reassembled[3*13::3] += np.random.uniform(-0.05, 0.05, size=len(sub2_opt)//3)
    v_reassembled[3*13+1::3] += np.random.uniform(-0.05, 0.05, size=len(sub2_opt)//3)
    v_reassembled[3*13+2::3] *= np.random.uniform(0.95, 1.05, size=len(sub2_opt)//3)

    # Phase 3: Local refinement with controlled perturbation
    perturbation = 0.03
    v_perturbed = v_reassembled.copy()
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
                             constraints=cons_perturbed, options={"maxiter": 500, "ftol": 1e-9})
    v = res_perturbed.x if res_perturbed.success else v_reassembled

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())