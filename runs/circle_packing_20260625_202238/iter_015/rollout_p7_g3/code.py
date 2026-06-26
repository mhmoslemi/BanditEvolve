import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    xs = (np.arange(n) % cols + 0.5) / cols
    ys = (np.arange(n) // cols + 0.5) / cols
    r0 = 0.5 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = xs
    v0[1::3] = ys
    v0[2::3] = r0

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

    # Initial optimization
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Compute constraint tightness
    constraint_tightness = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < v[3*i+2] + v[3*j+2] - 1e-5:
                constraint_tightness[i] += (v[3*i+2] + v[3*j+2] - dist)
                constraint_tightness[j] += (v[3*i+2] + v[3*j+2] - dist)

    # Sort indices by constraint tightness (most constrained first)
    sorted_indices = np.argsort(constraint_tightness)
    
    # Permute the decision vector based on sorted indices
    permuted_v = np.zeros_like(v)
    for i, idx in enumerate(sorted_indices):
        permuted_v[3*i] = v[3*idx]
        permuted_v[3*i+1] = v[3*idx+1]
        permuted_v[3*i+2] = v[3*idx+2]
    
    # Re-optimize with permuted initial guess
    res = minimize(neg_sum_radii, permuted_v, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else permuted_v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Add penalty for overlapping circles to improve convergence
    def penalty(v):
        sum_penalty = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < v[3*i+2] + v[3*j+2] - 1e-5:
                    sum_penalty += max(0, (v[3*i+2] + v[3*j+2] - dist) ** 2)
        return sum_penalty

    # Re-optimize with penalty
    res = minimize(lambda v: -np.sum(v[2::3]) + 100 * penalty(v), v, method="SLSQP", 
                   bounds=bounds, constraints=cons, 
                   options={"maxiter": 500, "ftol": 1e-9})
    v = res.x if res.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)

    # Final cleanup pass: attempt to infinitesimally inflate circles without moving centers
    # Create a copy of the current solution
    cleanup_v = v.copy()
    # Attempt to increase each radius by a small amount, keeping centers fixed
    for i in range(n):
        # Check if we can increase the radius without violating constraints
        dx = cleanup_v[3*i] - cleanup_v[3*i+2]
        dy = cleanup_v[3*i+1] - cleanup_v[3*i+2]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist > cleanup_v[3*i+2] + 1e-5:
            # We can try to increase the radius
            cleanup_v[3*i+2] += 1e-4
            cleanup_v[3*i+2] = np.clip(cleanup_v[3*i+2], 1e-4, 0.5)
    
    # Re-optimize with the cleanup initial guess
    res_cleanup = minimize(neg_sum_radii, cleanup_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
    cleanup_v = res_cleanup.x if res_cleanup.success else cleanup_v
    
    # Validate the cleanup solution
    valid, msg = validate_packing(np.column_stack([cleanup_v[0::3], cleanup_v[1::3]]),
                                  cleanup_v[2::3])
    if valid:
        v = cleanup_v
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = np.clip(v[2::3], 1e-6, None)
    
    return centers, radii, float(radii.sum())