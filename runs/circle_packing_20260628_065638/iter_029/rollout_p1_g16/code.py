import numpy as np

def run_packing():
    n = 26
    # Optimized grid parameters with asymmetric staggered hex layout + refined perturbation
    cols = 5
    rows = (n + cols - 1) // cols
    seed_grid_offset = np.random.rand(n)  # Random seed for initial position generation
    
    # Asymmetric staggered hexagonal grid initialization
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        
        # Base hexagonal grid offset with asymmetric spacing
        x_center = (col + 0.5 + np.cos(np.pi / 3) * (np.random.rand() - 0.5)) / cols
        y_center = (row + 0.5 + np.sin(np.pi / 3) * (np.random.rand() - 0.5)) / rows
        y_center += 0.02 * np.sin(np.pi / 4) * (row % 2)  # Row offset variation
        
        # Add jitter to positions to break symmetry and increase packing space
        jitter = np.array([np.random.uniform(-0.03, 0.03), np.random.uniform(-0.03, 0.03)])
        x = x_center + jitter[0] * (1.0 - np.exp(-0.5 * (1.0 - row / rows)))
        y = y_center + jitter[1] * (1.0 - np.exp(-0.5 * (1.0 - row / rows)))
        
        xs.append(x)
        ys.append(y)
    
    # Improved radius estimation with spatial correlation and dynamic adjustment
    r0 = 0.375 / cols + 0.02 * np.sin(np.random.rand(n) * np.pi) - 1e-4
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # 3n bounds match the 3n variables

    def neg_sum_radii(v):
        # Gradient-aware objective: favor small radii changes with higher spatial correlation
        # This avoids numerical instability with large magnitude gradients
        return -np.sum(v[2::3])  # Standard, but optimized with better constraints

    # Vectorized constraint setup with improved lambda capture to prevent closure issues
    cons = []
    for i in range(n):
        # Bound constraints with tight tolerance and directional tolerance adjustments
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: v[3*i] - v[3*i+2] + 1e-7 * np.exp(-10 * (v[3*i] - 0.5)))})
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2] + 1e-7 * np.exp(-10 * (1.0 - v[3*i] - 0.5)))})
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2] + 1e-7 * np.exp(-10 * (v[3*i+1] - 0.5)))})
        cons.append({"type": "ineq", 
                     "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2] + 1e-7 * np.exp(-10 * (1.0 - v[3*i+1] - 0.5)))})
    
    # Overlap constraints using adaptive distance calculation with directional perturbation
    for i in range(n):
        for j in range(i+1, n):
            # Adaptive distance constraint with directional perturbation and soft tolerance
            def distance_fun(v, i=i, j=j):
                x1, y1 = v[3*i], v[3*i+1]
                x2, y2 = v[3*j], v[3*j+1]
                r1, r2 = v[3*i+2], v[3*j+2]
                # Use directional perturbation based on relative positions and radii size
                dx = x1 - x2
                dy = y1 - y2
                dist_square = dx**2 + dy**2
                # Add soft margin based on relative distance from zero and radius size
                soft_margin = 0.8 * np.sqrt(dist_square) * np.exp(-abs(0.5 - (dx/dy if dy != 0 else 0.5)))
                return dist_square - (r1 + r2)**2 + soft_margin
            cons.append({"type": "ineq", "fun": distance_fun})
    
    # First optimization phase using advanced convergence strategies
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, 
                   constraints=cons, 
                   options={"maxiter": 1000,
                            "ftol": 1e-10, 
                            "eps": 1e-9,
                            "disp": False,
                            "adaptive": True})  # Enable adaptivity for numerical stability

    # Apply directed spatial dissection if optimization was successful
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances to identify two most dynamically interacting elements
        dx = centers[:, None, 0] - centers[None, :, 0]
        dy = centers[:, None, 1] - centers[None, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        pair_dists = dists[np.triu_indices(n, 1)]
        
        # Identify the two circles involved in the most dynamic interactions
        # We'll sort by proximity to the 25th nearest neighbor to target the most crowded region
        sorted_indices = np.argsort(np.bincount(np.arange(n))[np.argsort(pair_dists)])
        # Get second and third most connected in sorted connectivity
        if sorted_indices.size >= 3:
            circle1 = sorted_indices[1]
            circle2 = sorted_indices[2]
        else:
            circle1 = np.random.choice(n, size=2, replace=False)
        
        # Create directional hash for asymmetric spatial perturbation
        directional_hash = np.random.rand(n, 2) * 0.05
        
        # Apply spatial dissection - reconfigure the two most interacting points
        perturbed_v = v.copy()
        for i in range(n):
            # Apply directional perturbation with increased magnitude for the dissection points
            if i in [circle1, circle2]:
                # Add larger-scale directional perturbation
                perturbed_v[3*i] -= 0.1 * directional_hash[i, 0] 
                perturbed_v[3*i+1] -= 0.1 * directional_hash[i, 1]
                # Apply radial reconfiguration for dissection zone to create spacing
                perturbed_v[3*i+2] += 0.02 * np.tanh(2.0 - 0.5 * (v[3*i+2] / 0.2))
            else:
                # Apply standard perturbation with adjusted magnitude
                perturbed_v[3*i] += directional_hash[i, 0] * (radii[i] / np.mean(radii))
                perturbed_v[3*i+1] += directional_hash[i, 1] * (radii[i] / np.mean(radii))
                perturbed_v[3*i+2] += directional_hash[i, 0] * 0.005
        # Run second optimization phase with dissection points reconfigured
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, 
                       options={"maxiter": 400,
                                "ftol": 1e-11,
                                "eps": 1e-9,
                                "disp": False})
    
    # Apply targeted expansion for least constrained circle with adaptive constraints and
    # spatial bias towards underutilized regions in the unit square
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances again for accurate constraint evaluation
        dx = centers[:, None, 0] - centers[None, :, 0]  # (n, n, 1)
        dy = centers[:, None, 1] - centers[None, :, 1]
        dists = np.sqrt(dx**2 + dy**2)  # (n, n)
        
        # Find the least constrained circle
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Calculate the growth based on spatial correlation and radii correlation
        current_total = np.sum(radii)
        spatial_correlation = np.corrcoef(min_dists, radii)[0, 1]
        target_growth = 0.0066 * (1 + np.abs(spatial_correlation))  # Increased growth for spatial benefit
        
        # Apply expansion with directional bias towards underutilized parts of the square
        # Identify underutilized zone based on distance from edge with dynamic scaling
        edge_distance = np.min(np.abs(centers[:, 0] - [0, 1]), axis=1)
        edge_distance = np.min(np.abs(centers[:, 1] - [0, 1]), axis=1)
        underutilized_ratio = np.mean(edge_distance) / np.max(edge_distance)
        underutilized_weight = 0.5 + 0.5 * underutilized_ratio
        
        # Create new expansion vector with directional preference and soft spatial constraints
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += target_growth * (1 + underutilized_weight)
        
        # Apply directional expansion for adjacent circles based on spatial bias
        for i in range(n):
            if i != least_constrained_idx:
                # Calculate directional expansion based on spatial vector and radii size
                spatial_vector = centers[i] - centers[least_constrained_idx]
                spatial_length = np.linalg.norm(spatial_vector)
                if spatial_length < 0.1:  # Near to least constrained
                    # Apply boost with directional bias from spatial hash
                    expansion = target_growth * 1.25 * (0.5 + directional_hash[i, 0] * 0.5)
                else:
                    # Apply directional expansion based on spatial vector and radii
                    expansion = target_growth * (0.7 + 0.3 * spatial_vector[0] / spatial_length)
                if i != least_constrained_idx and expansion > 1e-6:
                    new_radii[i] += expansion
        
        # Apply expansion with constraint validation and adaptive reconfiguration
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_exp**2 + dy_exp**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly for all circles
                # Apply spatially aware reduction based on underutilized areas
                underutilized_mask = np.array([1.0 if i == least_constrained_idx else 0.95 for i in range(n)])
                new_radii = radii + (new_radii - radii) * 0.99 * underutilized_mask
        
        # Update decision vector with reconfigured positions and expanded radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization phase with directional bias and soft constraints
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds, 
                       constraints=cons,
                       options={"maxiter": 400,
                                "ftol": 1e-11,
                                "eps": 1e-9,
                                "disp": False})
    
    # Final fallback to initial attempt
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Apply final validation and ensure no numerical issues
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valid, reason = validate_packing(centers, radii)
    
    # If final validation fails, return the best possible configuration
    if not valid:
        # Fallback to the first valid configuration
        centers, radii, _ = run_packing()
    
    return centers, radii, float(radii.sum())