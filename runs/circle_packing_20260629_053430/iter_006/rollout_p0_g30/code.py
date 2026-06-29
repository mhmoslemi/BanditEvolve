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

    # Phase 2: Structural decomposition and independent optimization of sub-components
    # Split into two sub-components: first 13 and last 13 circles
    sub1_centers = v[0::3][0:13]
    sub1_radii = v[2::3][0:13]
    sub2_centers = v[0::3][13:]
    sub2_radii = v[2::3][13:]

    # Optimize sub1 with perturbation
    perturbation = 0.05
    v_sub1_perturbed = v.copy()
    np.random.seed(42)
    for i in range(13):
        v_sub1_perturbed[3*i] += np.random.uniform(-perturbation, perturbation)
        v_sub1_perturbed[3*i+1] += np.random.uniform(-perturbation, perturbation)
        v_sub1_perturbed[3*i+2] += np.random.uniform(-perturbation, perturbation)

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

    res_sub1 = minimize(neg_sum_radii, v_sub1_perturbed, method="SLSQP", bounds=bounds_sub1,
                        constraints=cons_sub1, options={"maxiter": 500, "ftol": 1e-9})
    v_sub1 = res_sub1.x if res_sub1.success else v_sub1_perturbed

    # Optimize sub2 with perturbation
    v_sub2_perturbed = v.copy()
    np.random.seed(42)
    for i in range(13):
        v_sub2_perturbed[3*(i+13)] += np.random.uniform(-perturbation, perturbation)
        v_sub2_perturbed[3*(i+13)+1] += np.random.uniform(-perturbation, perturbation)
        v_sub2_perturbed[3*(i+13)+2] += np.random.uniform(-perturbation, perturbation)

    bounds_sub2 = []
    for _ in range(13):
        bounds_sub2 += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    cons_sub2 = []
    for i in range(13):
        cons_sub2.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+39] - v[3*i+41]})
        cons_sub2.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+39] - v[3*i+41]})
        cons_sub2.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+40] - v[3*i+41]})
        cons_sub2.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+40] - v[3*i+41]})
    for i in range(13):
        for j in range(i + 1, 13):
            def constraint_func_sub2(v, i=i, j=j):
                dx = v[3*(i+13)] - v[3*(j+13)]
                dy = v[3*(i+13)+1] - v[3*(j+13)+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*(i+13)+2] + v[3*(j+13)+2])**2
                return dist_sq - min_dist_sq
            cons_sub2.append({"type": "ineq", "fun": constraint_func_sub2})

    res_sub2 = minimize(neg_sum_radii, v_sub2_perturbed, method="SLSQP", bounds=bounds_sub2,
                        constraints=cons_sub2, options={"maxiter": 500, "ftol": 1e-9})
    v_sub2 = res_sub2.x if res_sub2.success else v_sub2_perturbed

    # Combine optimized sub-components
    v = np.concatenate([v_sub1, v_sub2])

    # Phase 3: Reassemble and finalize
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())