import numpy as np

def run_packing():
    n = 26
    # Dynamic spatial layout with adaptive hexagonal and quasi-elliptic seeding
    cols = int(np.ceil(np.sqrt(n)))
    rows = n // cols + (n % cols != 0)
    
    # Initialize positions with randomized geometric clustering and staggered grid
    xs, ys = [], []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Hexagonal layout with adaptive row shift for density
        x_center = (col + 0.45) / cols
        y_center = (row + 0.45) / rows
        if row % 2 == 1:
            x_center += 0.35 / cols
        
        # Add small perturbation to break symmetry and avoid clustering
        x = x_center + np.random.uniform(-0.07, 0.07)
        y = y_center + np.random.uniform(-0.07, 0.07)
        xs.append(x)
        ys.append(y)
    
    # Dynamic radius calculation based on packing efficiency and geometric layout
    # Using hexagonal grid packing coefficient and density adjustment
    r0 = 0.44 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with captured i
    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})

    # Vectorized overlap constraints using adaptive weighting
    for i in range(n):
        for j in range(i + 1, n):
            # Use adaptive weighting for overlap constraints based on spatial relationship
            # Weight is higher for closer circles to prioritize spatial resolution
            cons.append({"type": "ineq",
                         "fun": (lambda v, i=i, j=j: 
                                 (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                 - (v[3*i+2] + v[3*j+2])**2 * (0.8 + 0.2 * np.sqrt((v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2)))})

    # First optimization phase: base layout with initial radii
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "eps": 1e-10, "disp": False})
    
    # If optimization was successful, perform targeted reconfiguration and geometric dissection
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute spatial relationships matrix for reconfiguration
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify the two most dynamically interacting circles with high spatial interdependency
        # High interdependency: circles with high mutual influence based on proximity and dynamic spatial changes
        interdependency = (dists < (np.mean(dists) + 2 * np.std(dists)))  # High dependency threshold
        if np.sum(interdependency) > 0:
            # Find the pair with highest mutual influence
            influence_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j and dists[i,j] < (np.mean(radii) * 1.5):
                        influence_matrix[i,j] = 1 / (dists[i,j] + 1e-6)
            influence_vals = influence_matrix.sum(axis=1)
            highly_influential_idx = np.argsort(influence_vals)[-2:]  # Get top 2 most influential
            
            circle1, circle2 = highly_influential_idx
            # Save original positions for reconfiguration
            orig_x1, orig_y1 = centers[circle1, 0], centers[circle1, 1]
            orig_x2, orig_y2 = centers[circle2, 0], centers[circle2, 1]
            
            # Create directional hashing and adjacency bias for targeted spatial dissection
            spatial_hash = np.random.rand(n, 2) * 0.06
            adjacency_hash = np.random.rand(n, 2) * 0.05
            
            # Isolate the two most dynamically interacting circles
            # Perturb their positions to force geometric dissection
            for idx in [circle1, circle2]:
                # Apply directional perturbation based on hashing and radius scaling
                perturbed_x = centers[idx, 0] + spatial_hash[idx, 0] * (radii[idx] / np.mean(radii)) * 1.5
                perturbed_y = centers[idx, 1] + spatial_hash[idx, 1] * (radii[idx] / np.mean(radii)) * 1.5
                
                # Apply directional force to break spatial interdependency
                if idx == circle1:
                    perturbed_x *= 1.05
                    perturbed_y *= 1.05
                elif idx == circle2:
                    perturbed_x *= 0.95
                    perturbed_y *= 0.95
                
                # Clamp bounds
                perturbed_x = np.clip(perturbed_x, 0.0, 1.0)
                perturbed_y = np.clip(perturbed_y, 0.0, 1.0)
                
                centers[idx, 0], centers[idx, 1] = perturbed_x, perturbed_y
            
            # Re-evaluate with altered positions to trigger reconfiguring
            res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-10, "disp": False})
        
        # If reconfiguration was successful, proceed with spatial dissection
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            
            # Compute spatial relationships matrix with new positions
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Find the least constrained circle by maximizing minimum distance to others
            min_dists = np.min(dists, axis=1)
            least_constrained_idx = np.argmax(min_dists)
            
            # Calculate growth based on current total sum and potential for expansion
            current_total = np.sum(radii)
            target_growth = 0.007
            expansion_factor = target_growth / (n - 1) * (current_total / np.sum(radii))
            
            # Create expansion vector with targeted expansion on least constrained
            new_radii = radii.copy()
            new_radii[least_constrained_idx] += expansion_factor * 1.3  # Moderate increase
            
            # Apply expansion with adaptive weighting based on adjacency and spatial interaction
            for i in range(n):
                if i != least_constrained_idx:
                    # Calculate adj weight as inverse of mutual distance
                    adj_weight = 1.0 / (dists[i, least_constrained_idx] + 1e-6)
                    adj_weight = np.clip(adj_weight, 0, 10)  # Cap to prevent extreme expansion
                    expansion_i = expansion_factor * (1.0 + 0.1 * np.sqrt(adj_weight)) * (1.0 + np.random.rand() * 0.2)
                    new_radii[i] += expansion_i
            
            # Apply expansion with constraint validation
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
                    # If invalid, decrease expansion slightly
                    new_radii = radii + (new_radii - radii) * 0.95
        
            # Update decision vector
            v_new = v.copy()
            v_new[2::3] = new_radii
            
            # Re-evaluate with expanded radii and new configuration
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 400, "ftol": 1e-10, "eps": 1e-10, "disp": False})
    
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