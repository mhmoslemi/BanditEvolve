import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 0.5) / cols
        y = (row + 0.5) / rows
        # Randomized offset to break symmetry
        x += np.random.uniform(-0.05, 0.05)
        y += np.random.uniform(-0.05, 0.05)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.3 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Asymmetric reconfiguration: trigger randomized spatial constraint variation
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Randomize a subset of spatial constraints to trigger reconfiguration
        perturbation = np.random.rand(n, 2) * 0.05
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += perturbation[i, 0]
            perturbed_v[3*i+1] += perturbation[i, 1]
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
    
    # Targeted radius expansion: expand the least constrained circle with improved strategy
    if res.success:
        v = res.x
        centers = v[0::3], v[1::3]
        radii = v[2::3]
        # Compute constraint violations using vectorized operations
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        # Boundary violation
        boundary_violations = np.maximum(r - x, r - (1 - x), r - y, r - (1 - y))
        # Overlap violation
        dx = x.reshape(-1, 1) - x
        dy = y.reshape(-1, 1) - y
        dist = np.sqrt(dx*dx + dy*dy)
        overlap_violations = np.maximum(0, r.reshape(-1, 1) + r - dist)
        # Total constraint violation
        total_violations = np.sum(overlap_violations, axis=1) + boundary_violations
        # Find the least constrained circle
        least_constrained_idx = np.argmin(total_violations)
        # Expand its radius slightly and adjust its position to maintain feasibility
        # Apply bounded expansion to avoid numerical instability
        v[3*least_constrained_idx + 2] += np.clip(0.002, 0, 0.5 - r[least_constrained_idx])
        v[3*least_constrained_idx] += np.clip(0.005, 0, 1 - r[least_constrained_idx] - x[least_constrained_idx])
        v[3*least_constrained_idx+1] += np.clip(0.005, 0, 1 - r[least_constrained_idx] - y[least_constrained_idx])
        # Re-evaluate with adjusted parameters
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())