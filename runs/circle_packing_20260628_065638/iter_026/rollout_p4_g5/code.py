import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Generate random geometric hash for initial spatial configuration
    random_hash = np.random.rand(n, 2) * 0.05
    spatial_centers = np.zeros((n, 2))
    
    # Evenly distribute centers using geometric hashing
    for i in range(n):
        row = i // cols
        col = i % cols
        base_x = (col + 0.5) / cols
        base_y = (row + 0.5) / rows
        spatial_centers[i, 0] = base_x + random_hash[i, 0]
        spatial_centers[i, 1] = base_y + random_hash[i, 1]
    
    # Ensure staggered grid for alternating rows
    for i in range(n):
        row = i // cols
        if row % 2 == 1:
            spatial_centers[i, 0] += 0.5 / cols
    
    # Initialize radii based on spacing and hashing distortion
    max_dist = np.max(np.sqrt((spatial_centers[:, np.newaxis, 0] - spatial_centers[np.newaxis, :, 0])**2 +
                              (spatial_centers[:, np.newaxis, 1] - spatial_centers[np.newaxis, :, 1])**2))
    r0 = 0.3 / cols - 1e-3
    
    # Initialize decision vector
    v0 = np.empty(3 * n)
    v0[0::3] = spatial_centers[:, 0]
    v0[1::3] = spatial_centers[:, 1]
    v0[2::3] = np.full(n, r0)

    # Define bounds for positions and radii
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Total length 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraints to enforce square boundaries
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Vectorized pairwise overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
            cons.append({"type": "ineq", "fun": constraint_func})

    # Initial optimization with enhanced convergence settings
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "maxls": 200})

    # Topological reconfiguration by spatial hashing
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Create a spatial hash for radical reconfiguration
        spatial_hash = np.random.rand(n, 2) * 0.08
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
        
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "maxls": 150})

    # Targeted expansion of least constrained circle (smallest radius)
    if res.success:
        v = res.x
        radii = v[2::3].copy()
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized pairwise distance matrix for all circles
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)

        # Find circle with smallest radius and least interaction
        interaction_index = np.argmin(radii)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)

        # If smallest radius is least constrained, prioritize it
        if radii[interaction_index] <= radii[least_constrained_idx]:
            least_constrained_idx = interaction_index

        # Calculate expansion factor for radius
        target_total_sum = np.sum(radii) + 0.008
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)

        # Create expanded radii vector with soft enforcement
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.1  # Slight over-expansion
        for i in range(n):
            if i != least_constrained_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.05 * np.random.rand())

        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
                
            # Re-calculate centers
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
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
                # If invalid, decrease expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "maxls": 150})

    # Final optimization with enhanced exploration
    if res.success:
        v = res.x
        radii = v[2::3].copy()
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances for all circles
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circle with largest isolation metric (least interaction)
        interaction = np.sum(dists, axis=1)
        isolated_idx = np.argmax(interaction)
        
        # Calculate expansion factor for radius
        target_total_sum = np.sum(radii) + 0.008
        expansion_factor = (target_total_sum - np.sum(radii)) / (n - 1)
        
        # Create expanded radii vector with soft enforcement
        new_radii = radii.copy()
        new_radii[isolated_idx] += expansion_factor * 1.1  # Slight over-expansion
        for i in range(n):
            if i != isolated_idx:
                new_radii[i] += expansion_factor * (1.0 + 0.05 * np.random.rand())

        # Apply expansion with constraint validation
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
                
            # Re-calculate centers
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
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
                # If invalid, decrease expansion slightly
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector with new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Re-evaluate with expanded radii and new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "maxls": 150})

    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())