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

    # Phase 1: Initial optimization with a modified constraint order
    cons_phase1 = []
    for i in range(n):
        # Prioritize non-overlapping constraints over position bounds
        cons_phase1.append({"type": "ineq", "fun": lambda v, i=i: np.sum((v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2 for j in range(i+1, n))})
        cons_phase1.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_phase1.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_phase1.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_phase1.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    res_phase1 = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                         constraints=cons_phase1, options={"maxiter": 300, "ftol": 1e-9})
    
    # Phase 2: Local refinement with perturbed initial guess
    v_phase1 = res_phase1.x if res_phase1.success else v0
    v_phase2 = v_phase1.copy()
    # Introduce small perturbation to force exploration of new regions
    np.random.seed(42)
    perturbation = np.random.uniform(-0.01, 0.01, size=3*n)
    v_phase2 += perturbation
    
    # Refinement phase with tighter constraints and more iterations
    cons_phase2 = []
    for i in range(n):
        cons_phase2.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons_phase2.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons_phase2.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons_phase2.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons_phase2.append({"type": "ineq", "fun": constraint_func})

    res_phase2 = minimize(neg_sum_radii, v_phase2, method="SLSQP", bounds=bounds,
                         constraints=cons_phase2, options={"maxiter": 700, "ftol": 1e-10})
    
    v = res_phase2.x if res_phase2.success else v_phase1
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())