import numpy as np

def run_packing():
    n = 26
    cols = 5  # Hexagonal grid with 5 columns for better spacing
    rows = (n + cols - 1) // cols  # Ensure enough rows for 26 circles
    
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

    # Define constraints for boundaries
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

    # Two-stage optimization pipeline: first global optimization with SLSQP, then local refinement with L-BFGS-B
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10, "eps": 1e-8})
    v = res.x if res.success else v0

    # Local refinement with L-BFGS-B for tighter optimization
    bounds_local = bounds.copy()
    cons_local = cons.copy()
    
    # Perturb initial guess slightly to escape local minima
    v_local = v.copy()
    np.random.seed(42)
    for i in range(n):
        v_local[3*i] += np.random.uniform(-0.05, 0.05)
        v_local[3*i+1] += np.random.uniform(-0.05, 0.05)
        v_local[3*i+2] += np.random.uniform(-0.05, 0.05)

    res_local = minimize(neg_sum_radii, v_local, method="L-BFGS-B", bounds=bounds_local,
                         constraints=cons_local, options={"maxiter": 500, "ftol": 1e-10, "eps": 1e-8})
    v = res_local.x if res_local.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())