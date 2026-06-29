import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and adaptive spatial perturbation
    xs = []
    ys = []
    
    # Base grid calculation with adaptive row spacing
    base_grid = np.array([(col + 0.5) / cols for col in range(cols)])
    row_offsets = 0.0
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = base_grid[col]
        y_center = (row + 0.5) / rows + (row * 0.01)  # Add row height perturbation
        # Randomized offset to break symmetry, scaled by radius
        offset = np.random.uniform(-1.0, 1.0) * 0.2 * (0.5 / cols)
        x = x_center + offset
        y = y_center + offset
        # Shift alternating rows for staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    r0 = 0.4 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with explicit capture
    cons = []
    for i in range(n):
        # Left + radius <= 1.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0.0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
    
    # Vectorized overlap constraints using lambda with captured i,j
    for i in range(n):
        for j in range(i + 1, n):
            # Use broadcasting to avoid scalar lambda capture
            def _constraint_func(i, j):
                def _lambda(v):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                return _lambda
            cons.append({"type": "ineq", "fun": _constraint_func(i, j)})

    # Initial optimization with increased max iterations and tighter tolerance
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11})
    
    # Asymmetric reconfiguration: spatial constraint perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash with adaptive scaling for enhanced reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.07
        perturbed_v = v.copy()
        for i in range(n):
            # Perturb centers based on radius relative to average radius
            scale_factor = 1.0 + 0.1 * (radii[i] / np.mean(radii))
            perturbed_v[3*i] += spatial_hash[i, 0] * scale_factor
            perturbed_v[3*i+1] += spatial_hash[i, 1] * scale_factor
        
        # Re-evaluate with new spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    # Targeted radius expansion with smart constraint analysis
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find least constrained circle by maximizing minimum distance to others with distance-weighted penalty
        min_dists = np.min(dists, axis=1)
        # Use weighted min distance to avoid edge case where one circle is very small
        isolation_score = np.sum(1.0 / (dists + 1e-8), axis=1)
        least_constrained_idx = np.argmin(isolation_score)
        
        # Calculate growth based on current total sum and potential for expansion
        current_total = np.sum(radii)
        target_growth = 0.007  # Slightly more aggressive target than before
        expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
        
        # Create expansion with controlled radius growth and spatial rebalancing
        # Introduce spatial rebalancing to allow for expansion without causing new overlaps
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.3  # Slightly over-expanded to trigger
        for i in range(n):
            if i != least_constrained_idx:
                # Stochastic expansion with constraint-aware scaling
                expansion_i = expansion_factor * (1.0 + 0.2 * np.random.rand())
                new_radii[i] += expansion_i
        
        # Apply expansion with constraint validation to maintain feasibility
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration directly
            # Skip re-evaluating all constraints through optimizer for speed
            is_valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        is_valid = False
                        break
                if not is_valid:
                    break
            
            if is_valid:
                break
            else:
                # If invalid, back off expansion gradually
                # Use weighted backtracking to preserve most of previous expansion
                # Apply inverse scaling factor proportional to radius
                back_off = 0.96
                new_radii = radii + (new_radii - radii) * back_off
        
        # Update decision vector with validated expansion
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())