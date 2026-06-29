import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid with tighter bounds
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset with tighter range to prevent over-dispersal
        x = x_center + np.random.uniform(-0.04, 0.04)
        y = y_center + np.random.uniform(-0.04, 0.04)
        # Shift alternate rows to create staggered grid with smaller offset for better packing
        if row % 2 == 1:
            x += 0.25 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.38 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Radius bounds now explicitly fixed

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with lambda closure fix using partial
    from functools import partial
    cons = []
    for i in range(n):
        # Left constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": partial(lambda v, i: v[3*i] - v[3*i+2], i=i)})
        # Right constraint: 1.0 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": partial(lambda v, i: 1.0 - v[3*i] - v[3*i+2], i=i)})
        # Bottom constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": partial(lambda v, i: v[3*i+1] - v[3*i+2], i=i)})
        # Top constraint: 1.0 - y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": partial(lambda v, i: 1.0 - v[3*i+1] - v[3*i+2], i=i)})

    # Vectorized overlap constraints with lambda capture using partial
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1800, "ftol": 1e-11, "disp": False})
    
    # Asymmetric reconfiguration: apply stochastic spatial perturbation with adaptive magnitude
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial perturbation map with adaptive influence based on radius
        spatial_hash = np.random.rand(n, 2) * 0.03
        perturbed_v = v.copy()
        for i in range(n):
            scale = 0.5 * (radii[i] / np.mean(radii))
            perturbed_v[3*i] += spatial_hash[i, 0] * scale
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scale
        
        # Re-evaluate with new spatial configuration and tighter solver settings
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12, "disp": False})

    # Targeted radius expansion on least constrained circle with adaptive expansion
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Efficient distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find the circle with the largest minimum distance to all others
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate expansion factor with adaptive heuristic
        base_growth = 0.008
        expansion_factor = base_growth * (1 + 0.25 * np.random.rand())  # Stochastic expansion
        
        # Create new radii vector with expansion on least constrained circle and others
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.25  # Increased expansion for the best candidate
        for i in range(n):
            if i != least_constrained_idx:
                # Add expansion with some randomness to allow for flexibility
                new_radii[i] += expansion_factor * (1 + 0.2 * np.random.rand())

        # Validate and refine expanded radii with soft validation constraints
        iterations = 0
        while iterations < 2:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate distances between circles without exhaustive checking
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                break
            else:
                # If overlap detected, reduce expansion with weighted averaging
                adjustment = 0.95  # More aggressive reduction
                new_radii = radii + (new_radii - radii) * adjustment
                iterations += 1
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, expanded_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "disp": False})

    # Final configuration handling
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())