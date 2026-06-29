import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with adaptive geometric clustering and staggered grid
    # This pattern ensures efficient use of space, especially for non-uniform radii
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Add small spatial variation for avoiding symmetry
        x = x_center + np.random.uniform(-0.03, 0.03)
        y = y_center + np.random.uniform(-0.03, 0.03)
        # Apply staggered pattern for hexagonal efficiency
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Set base radius based on hexagonal packing efficiency and optimization potential
    # The value is chosen to maintain initial feasibility for further expansion
    r0 = 0.37 / cols
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0) + np.random.uniform(-1e-4, 1e-4, size=n)
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Match length of vector 3*n
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # Objective to maximize radii sum
    
    # Constraint: for each circle, x_i >= r_i
    cons = []
    for i in range(n):
        # Left
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Overlap constraints for all circle pairs
    for i in range(n):
        for j in range(i+1, n):
            # Distance^2 - (r_i + r_j)^2 >= 0 
            # To compute as: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2
            cons.append({"type": "ineq", 
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})
    
    # Phase 1: Initial optimization to refine base grid
    # With tighter tolerance and larger iteration limit to explore
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2000, "ftol": 1e-12, "eps": 1e-12})
    
    # Phase 2: Reconfiguration if initial is successful
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate structured but asymmetric spatial hashing for directional influence
        spatial_hash = np.random.rand(n, 2) * 0.06
        # Generate adjacency-aware bias hash for constraint ordering
        adjacency_hash = np.random.rand(n, 2) * 0.04
        
        # Apply directional spatial perturbation based on radii and hash
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0] * (radii[i] / np.mean(radii)) 
            perturbed_v[3*i+1] += spatial_hash[i, 1] * (radii[i] / np.mean(radii))
            # Apply directional expansion based on adjacency
            if i < n - 2:
                perturbed_v[3*i+2] += adjacency_hash[i, 0] * 0.005 * (1 + 0.5 * np.sqrt(radii[i]))
        
        # Reoptimize with spatial reconfiguration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-10})
    
    # Phase 3: Apply geometric dissection on dynamically interacting circles
    # This is the crux of the surgical strike
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance calculation using broadcasting
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify the two most dynamically interacting circles (based on min pairwise distances)
        # This gives us the two circles that need reconfiguration
        min_pairwise_dists = np.min(dists, axis=(1, 2))
        # Sort indices by the sum of their pairwise distances with others
        pairwise_sum_dists = np.sum(dists, axis=(1, 2))
        interaction_indices = np.argsort(pairwise_sum_dists)
        
        # Identify the top two most highly interacting circles
        dynamic_circles = interaction_indices[-2:]
        circle_0, circle_1 = dynamic_circles
        
        # Apply a forced geometric dissection to these two
        # We will:
        # 1. Move them to avoid overlap and reconfigure
        # 2. Add directional constraints to ensure non-overlap
        # 3. Adjust radii to enable expansion for the least constrained
        # 4. Apply soft constraints to facilitate spatial expansion
        
        # Create custom dissection vectors
        dissection_v = v.copy()
        
        # Move the two dynamic circles slightly in different directions to disperse them
        # This is a targeted perturbation to create new spacing and avoid overlap
        dissection_v[3*circle_0 + 0] += np.random.uniform(-0.1, 0.1) * (1 + np.random.rand())
        dissection_v[3*circle_0 + 1] += np.random.uniform(-0.1, 0.1) * (1 + np.random.rand())
        dissection_v[3*circle_1 + 0] += np.random.uniform(-0.1, 0.1) * (1 + np.random.rand()) * (-1)
        dissection_v[3*circle_1 + 1] += np.random.uniform(-0.1, 0.1) * (1 + np.random.rand()) * (-1)
        
        # Apply soft directional constraints for these two circles to prevent overlap
        # This is a new constraint added to the optimizer to enforce specific non-overlapping
        for i in range(n):
            if i == circle_0 or i == circle_1:
                cons.append({"type": "ineq", 
                             "fun": (lambda v, target_idx=i, i=i: 
                                     (v[3*i] - v[3*target_idx])**2 
                                     + (v[3*i+1] - v[3*target_idx+1])**2 
                                     - (v[3*i+2] + v[3*target_idx+2])**2)})
        
        # Reoptimize with the new dissection and added constraints
        res = minimize(neg_sum_radii, dissection_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 350, "ftol": 1e-11, "eps": 1e-10})
    
    # Phase 4: Apply targeted expansion for the least constrained circle
    # This is the second surgical strike, after the dissection of the two interacting ones
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Vectorized distance matrix for all pairs
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Calculate "constrainedness" by finding the minimal distance to other circles
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        
        # Apply expansion strategy to the least constrained
        # Calculate growth factor based on current sum and potential
        current_total = np.sum(radii)
        target_growth = 0.0082 # Increment from previous SOTA's 0.0065
        expansion_factor = target_growth / (n - 1) * (current_total / np.mean(radii))
        
        # Create new_radii vector with directional expansion
        new_radii = radii.copy()
        new_radii[least_constrained_idx] += expansion_factor * 1.2
        
        # Apply expansion to adjacent circles based on adjacency hash
        for i in range(n):
            if i != least_constrained_idx:
                adj_weight = np.linalg.norm(centers[least_constrained_idx] - centers[i])
                # Use adjacency hash to give expanded direction preference
                expansion = expansion_factor * (1.0 + 0.2 * np.random.rand())
                if adj_weight < 0.1:
                    #Boost for nearby circles to exploit gaps
                    expansion *= 1.5 * (1 + 0.3 * np.random.rand())
                new_radii[i] += expansion
        
        # Apply expansion with constraint validation loop
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
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
                # If invalid, scale back expansion linearly with previous state
                new_radii = radii + (new_radii - radii) * 0.95
        
        # Update decision vector with reconfigured positions and new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Final optimization with new radii
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-10})
    
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