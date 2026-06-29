import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Step 1: Initialize positions with advanced spatial perturbation and dynamic column adjustment
    xs = []
    ys = []
    np.random.seed(1234)  # Fixed seed for deterministic optimization steps
    
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Introduce adaptive spatial hashing with row-dependent perturbation
        # Row perturbation: increase for lower rows to spread vertically
        row_scaling = 1.0 + 0.2 * (row / rows)
        col_scaling = 1.0 + 0.1 * (row / rows)
        
        x = x_center + np.random.uniform(-0.04 * col_scaling, 0.04 * col_scaling)
        y = y_center + np.random.uniform(-0.04 * row_scaling, 0.04 * row_scaling)
        
        # Stagger row 2 and 4: alternate column shift to reduce vertical alignment issues
        if row % 3 in [1, 2]:
            x += 0.5 / cols * (0.5 * row_scaling)
        
        xs.append(x)
        ys.append(y)
    
    # Step 2: Dynamic radius initialization with spatial-aware adaptive scaling
    # Base radius is based on row spacing and col clustering
    base_radius = 0.40 / rows - 1e-2  # Better base radius than previous
    r0 = base_radius * np.ones(n)
    # Add spatial awareness: circles in "empty" rows get extra growth potential
    spatial_weights = 1.0 + 0.05 * (np.arange(n) // cols)  # Empty rows get larger spatial weights
    r0 = np.clip(r0 * spatial_weights, 1e-4, 0.4)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3*n entries
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Step 3: Optimize with vectorized boundary constraints and adaptive scaling
    # Constraint handling is fully vectorized and avoids lambda capture issues
    cons = []
    
    for i in range(n):
        # Left side constraint: x - r >= 0 => v[3*i] - v[3*i+2] >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Right side constraint: 1 - (x + r) >= 0 => 1 - v[3*i] - v[3*i+2] >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Bottom side constraint: y - r >= 0 => v[3*i+1] - v[3*i+2] >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Top side constraint: 1 - (y + r) >= 0 => 1 - v[3*i+1] - v[3*i+2] >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Step 4: Vectorized overlap constraints with spatial awareness
    for i in range(n):
        for j in range(i + 1, n):
            # Avoid direct use of lambda to prevent index capture issue
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx**2 + dy**2 - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Step 5: Core optimization with adaptive configuration
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 3000, "ftol": 1e-12, "eps": 1e-8})
    
    v = res.x if res.success else v0
    
    # Step 6: Multi-stage reconfiguration with spatial awareness and radius adjustment
    # First optimization pass
    if res.success:
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Step 6A: Geometric hashing and asymmetric spatial perturbation
        spatial_hash = np.random.rand(n, 2) * 0.04
        perturbation_factor = np.random.rand(n) * 1.2
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * radii[i] * 1.0
            perturbed_v[3*i+1] += spatial_hash[i, 1] * radii[i] * 1.0
            
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 600, "ftol": 1e-12, "eps": 1e-8})
    
    # Step 7: Radius expansion targeting least constrained circles with dynamic adjustment
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Step 7A: Vectorized distance matrix for constraint validation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Step 7B: Find the least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        min_available_dist = min_dists[least_constrained_idx]
        
        # Step 7C: Compute expansion factor with spatial awareness
        # Consider the available space for growth
        available_growth_space = (min_available_dist - radii[least_constrained_idx]) / 1.2
        expansion_factor = np.maximum(available_growth_space / (n - 1) * 1.1, 0.0001)
        
        # Step 7D: Expand radii with targeted and stochastic expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2
        for i in range(n):
            if i != least_constrained_idx:
                expansion_i = expansion_factor * (1.0 + 0.15 * np.random.rand())
                new_radii[i] += expansion_i
        
        # Step 7E: Validate and refine expanded configuration
        for _ in range(3):
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Step 7E1: Distance validation
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist_exp = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist_exp < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # Reduce expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Step 7F: Apply final configuration
        v_new = v.copy()
        v_new[2::3] = new_radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-8})
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())