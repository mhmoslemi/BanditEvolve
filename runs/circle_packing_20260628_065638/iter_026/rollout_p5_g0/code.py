import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and optimized staggered grid
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Use more precise offset to reduce clustering
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        offset_x = np.random.uniform(-0.06, 0.06)
        offset_y = np.random.uniform(-0.06, 0.06)
        x = x_center + offset_x
        y = y_center + offset_y
        # Shift alternate rows with a more dynamic staggering
        if row % 2 == 1:
            x += (np.random.uniform(0.2, 0.4) / cols)
        xs.append(x)
        ys.append(y)
    
    # Set initial radius with optimized spacing
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

    # Vectorized boundary constraints with improved lambda binding
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with optimized lambda closure
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with enhanced parameters
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-10})
    
    # Asymmetric reconfiguration: randomize spatial constraints with stochastic perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Apply asymmetric random spatial reconfiguration
        spatial_perturb = np.random.rand(n, 2) * 0.1
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_perturb[i, 0] * (radii[i] / np.mean(radii))
            perturbed_v[3*i+1] += spatial_perturb[i, 1] * (radii[i] / np.mean(radii))
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
    
    # Targeted radius expansion for the least constrained circle using spatial analysis
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation for all pairs
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate minimum distance for each circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        least_constrained_radii = radii[least_constrained_idx]
        
        # Calculate expansion factor with soft constraint
        target_growth = 0.007
        current_total = np.sum(radii)
        expansion_factor = target_growth / (n - 1) * (current_total - np.min(radii)) / current_total
        expansion_factor = max(expansion_factor, 0.001) if np.min(radii) > 0 else expansion_factor
        
        # Create new radii with expansion centered around the least constrained circle
        new_radii = radii.copy()
        # Add a base expansion to the least constrained circle
        new_radii[least_constrained_idx] += expansion_factor * 1.2
        # Add moderate expansion to others
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * np.random.uniform(0.8, 1.2)
        
        # Validate and refine expanded radii with local validation
        iterations = 0
        while iterations < 4:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate distances between all pairs
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
                # If overlap detected, reduce expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
                iterations += 1
        
        # Update decision vector with expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with new configuration to fine-tune
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())