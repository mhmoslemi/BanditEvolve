import numpy as np

def run_packing():
    n = 26
    
    # Initialize with hexagonal grid with asymmetric staggering 
    cols = 5  # Even number of columns ensures better stagger efficiency
    rows = (n + cols - 1) // cols
    xs = []
    ys = []
    
    # Use asymmetric hexagonal pattern with adaptive staggering
    for i in range(n):
        col = i % cols
        row = i // cols
        
        # Base hexagonal grid offset with asymmetric shift for better spacing
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        
        # Generate asymmetric horizontal shift for even rows
        shift = 0.0
        if row % 2 == 1 and col < cols - 1:
            shift = 0.4 / cols
        
        # Add small perturbations to avoid symmetry collapse
        x = x_center + np.random.uniform(-0.05, 0.05)
        y = y_center + np.random.uniform(-0.05, 0.05)
        x += shift
        xs.append(x)
        ys.append(y)
    
    # Radius estimation based on hexagonal packing with adaptive scaling
    r0 = 0.23 / cols  # Scaled down from previous to allow more flexibility
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Define bounds with exact matching of decision vector length
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)]
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Build constraints using closure with i and j
    cons = []
    for i in range(n):
        # Left wall constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right wall constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom wall constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top wall constraint
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Add inter-circle constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Add constraint for distance^2 - (r_i + r_j)^2 >= 0
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2)})
    
    # First phase: optimize base layout and refine
    res = minimize(neg_sum_radii, v0, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 2000, "ftol": 1e-11, 
                            "eps": 1e-10, "disp": False})
    
    # Targeted phase: isolate and reconfigure two most dynamically interacting circles
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Use vectorized distance matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        dists[np.arange(n), np.arange(n)] = np.inf
        
        # Find top two interacting pairs (minimum distance)
        min_distances = np.min(dists, axis=1)
        idx_to_remove = np.argsort(min_distances)[-2:]
        top_pair = np.array([idx_to_remove[0], idx_to_remove[1]])
        
        # Extract the two dynamically interacting circles for reconfiguration
        circle_1 = centers[top_pair[0]]
        circle_2 = centers[top_pair[1]]
        r1 = radii[top_pair[0]]
        r2 = radii[top_pair[1]]
        
        # Create perturbation vectors for directional movement and radius manipulation
        directional_vectors = np.random.rand(n, 2)
        directional_vectors *= 0.1
        directional_vectors[top_pair] *= 2  # Increase perturbation for interacting circles
        
        # Create adjacency hash for constraint-based expansion
        adjacency_hash = np.random.rand(n, 2)
        adjacency_hash *= 0.1
        
        # Create a perturbed configuration to break symmetry and reconfigure
        perturbed_v = v.copy()
        
        # Adjust positions of the two interacting circles with directional vectors
        # Apply scaled directional movement to break interdependencies
        for idx in top_pair:
            perturbed_v[3*idx] += directional_vectors[idx, 0] * (r1 if idx == top_pair[0] else r2)
            perturbed_v[3*idx+1] += directional_vectors[idx, 1] * (r1 if idx == top_pair[0] else r2)
        
        # Apply adjacency-based radius expansion to nearby circles
        for idx in range(n):
            if idx not in top_pair:
                # Apply directional radius expansion with adjacency weighting
                adj_weight = np.linalg.norm(centers[top_pair[0]] - centers[idx])
                if adj_weight < 0.1:
                    # Boost expansion for nearby circles
                    expansion = np.random.uniform(0.001, 0.003)
                else:
                    expansion = np.random.uniform(0.0, 0.001)
                perturbed_v[3*idx+2] += expansion * (1.0 + adjacency_hash[idx, 0])

        # Re-optimization for spatial reconfiguration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 800, "ftol": 1e-11, 
                                "eps": 1e-10, "disp": False})
        
        # If optimization improved the configuration, proceed to final expansion
        if res.success:
            v = res.x
            new_centers = np.column_stack([v[0::3], v[1::3]])
            new_radii = v[2::3]
            
            # Recompute vectorized distance matrix for validation
            dx = new_centers[:, np.newaxis, 0] - new_centers[np.newaxis, :, 0]
            dy = new_centers[:, np.newaxis, 1] - new_centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            dists[np.arange(n), np.arange(n)] = np.inf
            
            # Compute least constrained circle based on current minimum distance
            min_dists = np.min(dists, axis=1)
            least_constrained_idx = np.argmax(min_dists)
            
            # Compute expansion vector based on adjacency and current spacing
            expansion_factor = 0.007 / (n - 1)
            current_total = np.sum(new_radii)
            base_growth = expansion_factor * current_total / np.mean(new_radii)
            expansion_vector = np.full(n, base_growth * 0.8)
            
            # Apply directional expansion to least constrained circle
            expansion_vector[least_constrained_idx] *= 1.4
            for i in range(n):
                if i != least_constrained_idx:
                    adj_weight = np.linalg.norm(new_centers[least_constrained_idx] - new_centers[i])
                    if adj_weight < 0.1:
                        expansion_vector[i] *= 1.1
                    else:
                        expansion_vector[i] *= 1.05
            
            # Apply expansion with gradient constraint validation
            while True:
                expanded_v = v.copy()
                expanded_v[2::3] = new_radii + expansion_vector
                expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
                
                # Validate expanded configuration
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        dx_exp = expanded_centers[i, 0] - expanded_centers[j, 0]
                        dy_exp = expanded_centers[i, 1] - expanded_centers[j, 1]
                        dist = np.sqrt(dx_exp**2 + dy_exp**2)
                        if dist < (new_radii[i] + expansion_vector[i] + new_radii[j] + expansion_vector[j]) - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    break
                else:
                    # If invalid, decrease expansion by small factor
                    expansion_vector *= 0.95
            
            # Update the decision vector with expanded radii
            v_final = v.copy()
            v_final[2::3] = new_radii + expansion_vector
            
            # Final fine-tuning optimization
            res = minimize(neg_sum_radii, v_final, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"maxiter": 800, "ftol": 1e-11, 
                                    "eps": 1e-10, "disp": False})
    
    # Final fallback to initial attempt
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    
    # Final validation to catch any potential numerical errors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valid, reason = validate_packing(centers, radii)
    
    if not valid:
        # Fallback to best possible valid configuration
        centers, radii, _ = run_packing()
    
    return centers, radii, float(radii.sum())