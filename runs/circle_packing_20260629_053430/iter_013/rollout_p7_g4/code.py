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

    # Vectorized overlap constraint function with modified distance function
    def vectorized_overlap_constraint(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
        return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)

    # Build constraints
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Add overlap constraints using vectorized function
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dist_sq = (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)
            cons.append({"type": "ineq", "fun": constraint_func})

    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10, "gtol": 1e-9})
    v = res.x if res.success else v0

    # Phase 2: Apply nonlinear coordinate warping
    warp_factor = 1.1
    scaled_v = v.copy()
    scaled_v[0::3] *= warp_factor
    scaled_v[1::3] *= warp_factor
    scaled_v[1::3] += 0.1  # slight vertical shift to break symmetry
    scaled_v[2::3] *= warp_factor

    bounds_warp = []
    for _ in range(n):
        bounds_warp += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    cons_warp = []
    for i in range(n):
        cons_warp.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_warp.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_warp.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_warp.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func_warp(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)
            cons_warp.append({"type": "ineq", "fun": constraint_func_warp})

    res_warp = minimize(neg_sum_radii, scaled_v, method="SLSQP", bounds=bounds_warp,
                       constraints=cons_warp, options={"maxiter": 600, "ftol": 1e-10, "gtol": 1e-9})
    v = res_warp.x if res_warp.success else v

    # Phase 3: Targeted radius perturbation to smallest circles
    smallest_indices = np.argsort(v[2::3])[:3]  # Select 3 smallest circles
    for idx in smallest_indices:
        v[3*idx + 2] *= 1.1  # Increase radius of smallest circles
    
    bounds_perturb = []
    for _ in range(n):
        bounds_perturb += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    cons_perturb = []
    for i in range(n):
        cons_perturb.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_perturb.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_perturb.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_perturb.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func_perturb(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                return np.log(dist_sq + 1e-12) - np.log(min_dist_sq + 1e-12)
            cons_perturb.append({"type": "ineq", "fun": constraint_func_perturb})

    res_perturb = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds_perturb,
                          constraints=cons_perturb, options={"maxiter": 800, "ftol": 1e-10, "gtol": 1e-9})
    v = res_perturb.x if res_perturb.success else v

    # Phase 4: Final refinement with controlled perturbation
    perturbation = 0.02
    v_final = v.copy()
    np.random.seed(42)
    for i in range(n):
        v_final[3*i] += np.random.uniform(-perturbation, perturbation)
        v_final[3*i+1] += np.random.uniform(-perturbation, perturbation)
        v_final[3*i+2] += np.random.uniform(-perturbation, perturbation)

    bounds_final = []
    for _ in range(n):
        bounds_final += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    res_final = minimize(neg_sum_radii, v_final, method="SLSQP", bounds=bounds_final,
                        constraints=cons_perturb, options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-9})
    v = res_final.x if res_final.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())