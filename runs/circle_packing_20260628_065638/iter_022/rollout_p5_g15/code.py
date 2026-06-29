import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        # Shift alternate rows to create staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
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
    
    # Vectorized overlap constraints with geometric hashing
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
    
    # Trigger constrained reconfiguration: randomized spatial constraint function
    if res.success:
        v = res.x
        # Stochastic spatial perturbation for exploration
        random_perturb = np.random.rand(n, 2) * 0.1
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_perturb[i, 0] * 1.5
            perturbed_v[3*i+1] += random_perturb[i, 1] * 1.5
        
        # Re-evaluate with perturbed parameters
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion on most isolated circle with adjacency-aware expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        # Find most isolated circle using distance-based isolation metric
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Use reciprocal of minimum distance to identify isolation
        min_dists = np.min(dists, axis=1)
        isolation = 1 / (min_dists + 1e-8)
        isolated_idx = np.argmin(isolation)
        
        # Compute current total sum
        total_sum = np.sum(radii)
        # Calculate expansion factor for controlled radius increase
        target_total_sum = total_sum + 0.008
        expansion_factor = (target_total_sum - total_sum) / (n - 1)
        
        # Create adjusted radius vector with adjacency-aware expansion
        new_radii = radii.copy()
        # Prioritize least-constrained circle
        new_radii[isolated_idx] += expansion_factor * 1.2
        # Distribute expansion to others, maintaining constraints
        new_radii += expansion_factor * (1 - (np.arange(n) == isolated_idx))
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and tightened tolerances
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12})

    # Final optimization pass with tighter tolerances and constraints
    if res.success:
        v = res.x
        # Final adjustment to stabilize configuration
        radii = v[2::3]
        # Recompute distances for final constraint check
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        # Ensure all constraints are satisfied
        constraint_values = np.zeros((n*(n-1)//2,))
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                constraint_values[idx] = dists[i,j] - (radii[i] + radii[j])
                idx += 1
        # If any constraint is too tight, reduce radii slightly
        tight_constraints = constraint_values < -1e-8
        if np.any(tight_constraints):
            for i in range(n):
                for j in range(i+1, n):
                    if tight_constraints[idx]:
                        radii[i] *= 0.999
                        radii[j] *= 0.999
                        idx += 1
        # Update v with adjusted radii
        v[2::3] = radii
        
        # Final optimization with minimal perturbation
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-12})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())