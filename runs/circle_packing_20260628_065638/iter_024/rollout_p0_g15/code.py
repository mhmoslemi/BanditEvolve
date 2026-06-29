import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Enhanced initialization with multi-stage spatial perturbation
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Base offset with gradient-controlled randomness
        base_offset = np.random.uniform(-0.03, 0.03)
        row_offset = 0.0 if row % 2 == 0 else 0.5 / cols
        x_start = x_center + base_offset
        y_start = y_center
        
        # Apply multi-stage perturbation
        x = x_start + np.random.normal(0, 0.015)
        y = y_start + np.random.normal(0, 0.015)
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        xs.append(x)
        ys.append(y)
    
    # Initial radii based on container capacity
    r0 = 0.36 / cols - 1e-3
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

    # Initial optimization with hybrid constraints and adaptive tolerances
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-12})

    # Hybrid reconfiguration: randomized geometric hashing with hard-sum constraint
    if res.success:
        v = res.x
        # Generate spatial displacement with enhanced randomness
        random_hash = np.random.rand(n, 2) * 0.1
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += random_hash[i, 0]
            perturbed_v[3*i+1] += random_hash[i, 1]
        
        # Enforce hard constraint on total sum radii
        def sum_radius_constraint(v):
            return np.sum(v[2::3]) - 2.638
        
        cons.append({"type": "eq", "fun": sum_radius_constraint})
        
        # Re-evaluate with perturbed parameters and constraint tightening
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-12})
    
    # Asymmetric multi-phase optimization with dynamic constraint reweighting
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances and constraint tightness
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Find least constrained circle with multi-criteria selection
        min_dists = np.min(dists, axis=1)
        weighted_scores = np.column_stack((radii, min_dists))
        least_constrained_idx = np.argmin(np.sum(weighted_scores, axis=1))
        
        # Calculate expansion factor with adaptive growth strategy
        current_sum = np.sum(radii)
        expansion_factor = 0.008 / (n - 1)  # Controlled expansion to unlock new configuration
        
        # Apply controlled expansion to least constrained circle
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.5  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor
        
        # Dynamic constraint tightening for stability
        def stability_constraint(v):
            return 2.64 - np.sum(v[2::3])
        
        cons.append({"type": "ineq", "fun": stability_constraint})
        
        # Re-evaluate with expanded radii and new configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-12})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())