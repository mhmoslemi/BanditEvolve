import numpy as np

def run_packing():
    n = 26
    cols = 5  # 5 columns for 26 circles, 6 rows
    rows = (n + cols - 1) // cols
    
    # Initial positions with randomized offset and row-staggering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add small random offset to break symmetry
        x = x_center + np.random.uniform(-0.1, 0.1)
        y = y_center + np.random.uniform(-0.1, 0.1)
        # Stagger alternate rows to improve spacing
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initial radii based on minimum spacing with padding
    r0 = 0.38 / cols - 1e-3  # Slightly improved from 0.36
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds consistent with 3*n parameters
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Objective to maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraint functions with fixed lambda capture
    def create_boundary_constraints(i):
        def fun_left(v):
            return 1.0 - v[3*i] - v[3*i+2]
        def fun_right(v):
            return v[3*i] - v[3*i+2]
        def fun_bottom(v):
            return 1.0 - v[3*i+1] - v[3*i+2]
        def fun_top(v):
            return v[3*i+1] - v[3*i+2]
        return fun_left, fun_right, fun_bottom, fun_top

    cons = []
    for i in range(n):
        fl, fr, fb, ft = create_boundary_constraints(i)
        cons.append({"type": "ineq", "fun": fl})
        cons.append({"type": "ineq", "fun": fr})
        cons.append({"type": "ineq", "fun": fb})
        cons.append({"type": "ineq", "fun": ft})
    
    # Overlap constraints using lambda captures with fixed i,j
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with tighter tolerances and more iterations
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2300, "ftol": 1e-12, "eps": 1e-10})
    
    # Asymmetric reconfiguration: perturb centers with random geometric hash
    if res.success:
        v = res.x
        # Apply random geometric hashing for spatial perturbation
        spatial_hash = np.random.rand(n, 2) * 0.06
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-10})
        
        # Targeted radius expansion strategy
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Calculate isolation metric: sum of inverse distances
            # This favors circles with more empty space around them
            inv_dists = 1.0 / (dists + 1e-10)
            isolation = np.sum(inv_dists, axis=1)
            least_constrained_idx = np.argmax(isolation)  # Most isolated circle
        
            # Apply targeted expansion with spatial-awareness
            max_allowed = 0.5
            min_radius = 1e-4
            current_total = np.sum(radii)
            target_total = current_total + 0.015  # Incremental improvement
            expansion_base = (target_total - current_total) / (n - 1)
            
            # Initialize new radii vector
            new_radii = radii.copy()
            for i in range(n):
                if i == least_constrained_idx:
                    # Expand the most isolated circle first
                    max_possible = max_allowed - radii[i]
                    new_radii[i] = np.clip(radii[i] + expansion_base * 1.2, min_radius, max_allowed)
                else:
                    # Moderate expansion with decay toward edge circles
                    expansion_factor = expansion_base * (1.0 - 0.1 * np.random.rand())
                    new_radii[i] = np.clip(radii[i] + expansion_factor, min_radius, max_allowed)
            
            # Reoptimize with new radii while maintaining constraints
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            res_expanded = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                                   constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-10})
            
            if res_expanded.success:
                v = res_expanded.x
    
    # Final validation and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())