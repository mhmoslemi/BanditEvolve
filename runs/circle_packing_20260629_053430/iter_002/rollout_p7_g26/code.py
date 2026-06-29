import numpy as np

def run_packing():
    n = 26
    cols = 5  # Manual adjustment for a hexagonal grid
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

    # Phase 1: Global search with relaxed constraints
    res1 = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-8, "eps": 1e-8})
    v1 = res1.x if res1.success else v0

    # Phase 2: Local refinement with tighter constraints and perturbation
    v2 = v1.copy()
    # Apply controlled perturbation to seed configuration
    perturb = np.random.normal(0, 0.01, size=v2.shape)
    v2 += perturb
    v2 = np.clip(v2, 0, 1)

    # Create refined constraints with tighter bounds
    refined_bounds = []
    for _ in range(n):
        refined_bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    res2 = minimize(neg_sum_radii, v2, method="SLSQP", bounds=refined_bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9, "eps": 1e-9})
    v = res2.x if res2.success else v1

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())