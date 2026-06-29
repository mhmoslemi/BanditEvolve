import numpy as np

def run_packing():
    n = 26
    cols = 5  # Fixed hexagonal grid with 5 columns for better spacing
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

    # First optimization: global search with enhanced constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    v = res.x if res.success else v0

    # Second phase: prioritize expansion of smallest circles
    # Sort indices by current radius to focus on smallest circles
    sorted_indices = np.argsort(v[2::3])
    v_sorted = np.copy(v)
    for i in range(n):
        v_sorted[3*sorted_indices[i]] = v[3*sorted_indices[i]]
        v_sorted[3*sorted_indices[i]+1] = v[3*sorted_indices[i]+1]
        v_sorted[3*sorted_indices[i]+2] = v[3*sorted_indices[i]+2]
    
    # Perform localized expansion of smallest circles
    perturbation = np.random.rand(3 * n) * 0.05
    v_expanded = v_sorted + perturbation
    v_expanded = np.clip(v_expanded, 0.0, 1.0)
    v_expanded[2::3] = np.clip(v_expanded[2::3], 1e-4, 0.5)

    # Third phase: refine with additional constraints and jitter
    res_expanded = minimize(neg_sum_radii, v_expanded, method="SLSQP", bounds=bounds,
                            constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res_expanded.x if res_expanded.success else v_expanded

    # Final refinement with random jitter
    perturbation = np.random.rand(3 * n) * 0.05
    v_final = v + perturbation
    v_final = np.clip(v_final, 0.0, 1.0)
    v_final[2::3] = np.clip(v_final[2::3], 1e-4, 0.5)

    # Final optimization with refined starting point
    res_final = minimize(neg_sum_radii, v_final, method="SLSQP", bounds=bounds,
                        constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res_final.x if res_final.success else v_final

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())