import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))  # Dynamic column count based on square root
    rows = (n + cols - 1) // cols  # Calculate rows based on columns

    # Initialize positions using a hexagonal grid pattern
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Offset even rows for hexagonal packing
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

    # Add overlap constraints with penalty function for smoother optimization
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (v[3*i+2] + v[3*j+2])**2
                # Allow small overlap with penalty
                return dist_sq - min_dist_sq + 1e-4 * max(0, (v[3*i+2] + v[3*j+2] - np.sqrt(dist_sq)))
            cons.append({"type": "ineq", "fun": constraint_func})

    # First phase: global optimization with SLSQP
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "eps": 1e-8})
    v = res.x if res.success else v0

    # Second phase: local refinement with L-BFGS-B
    res_local = minimize(neg_sum_radii, v, method="L-BFGS-B", bounds=bounds,
                         constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res_local.x if res_local.success else v

    # Third phase: perturbation and mutation strategy
    np.random.seed(42)
    perturbation = np.random.uniform(-0.05, 0.05, size=3*n)
    v_perturbed = v + perturbation
    v_perturbed = np.clip(v_perturbed, 0.0, 1.0)
    v_perturbed[2::3] = np.clip(v_perturbed[2::3], 1e-4, 0.5)

    # Refine the perturbed solution with local optimization
    res_perturbed = minimize(neg_sum_radii, v_perturbed, method="L-BFGS-B", bounds=bounds,
                             constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res_perturbed.x if res_perturbed.success else v

    # Fourth phase: advanced mutation and reordering
    np.random.seed(43)
    indices = np.random.permutation(n)
    v_mutated = v.copy()
    for i in range(n):
        v_mutated[3*indices[i]] = v[3*i]
        v_mutated[3*indices[i]+1] = v[3*i+1]
        v_mutated[3*indices[i]+2] = v[3*i+2]
    v_mutated = np.clip(v_mutated, 0.0, 1.0)
    v_mutated[2::3] = np.clip(v_mutated[2::3], 1e-4, 0.5)

    # Refine the mutated solution with local optimization
    res_mutated = minimize(neg_sum_radii, v_mutated, method="L-BFGS-B", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res_mutated.x if res_mutated.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())