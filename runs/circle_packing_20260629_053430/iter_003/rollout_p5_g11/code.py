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
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # First phase: global optimization with initial layout
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Second phase: expand smallest circles first
    # Sort circles by radius to prioritize expansion of smallest ones
    indices = np.argsort(v[2::3])
    sorted_v = v.copy()
    sorted_v[2::3] = v[2::3][indices]
    sorted_v[0::3], sorted_v[1::3] = v[0::3][indices], v[1::3][indices]
    
    # Optimization phase 2: expand smallest circles only
    def neg_sum_radii_expanded(v):
        return -np.sum(v[2::3])
    
    res_expanded = minimize(neg_sum_radii_expanded, sorted_v, method="SLSQP", bounds=bounds,
                            constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v_expanded = res_expanded.x if res_expanded.success else sorted_v

    # Third phase: random jitter to escape local optima and explore new configurations
    perturbation = np.random.rand(3 * n) * 0.05
    v_perturbed = v_expanded + perturbation
    v_perturbed = np.clip(v_perturbed, 0.0, 1.0)
    v_perturbed[2::3] = np.clip(v_perturbed[2::3], 1e-4, 0.5)

    # Optimization phase 3: refine with perturbed starting point
    res_perturbed = minimize(neg_sum_radii, v_perturbed, method="SLSQP", bounds=bounds,
                             constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res_perturbed.x if res_perturbed.success else v_expanded

    # Final refinement to ensure all constraints are met
    res_final = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                         constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res_final.x if res_final.success else v

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())