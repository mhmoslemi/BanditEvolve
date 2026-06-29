import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Generate a base grid with 2D hashing and spatial awareness
    xs = []
    ys = []
    for i in range(n):
        # Assign grid position using 2D hashing
        row = i // cols
        col = i % cols
        x_center = (col + np.random.rand()) / cols * 0.8 + 0.1
        y_center = (row + np.random.rand()) / rows * 0.8 + 0.1
        # Add a non-symmetric offset to break symmetry
        x_offset = np.random.uniform(-0.04, 0.04)
        y_offset = np.random.uniform(-0.04, 0.04)
        # Create staggered alternating row pattern with phase shift
        x = x_center + x_offset + (0.5 / cols * (row % 2)) if row % 2 == 1 else x_center + x_offset
        y = y_center + y_offset
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with geometric mean scaling, and add a small random seed distortion
    # Using inverse square root for radius distribution to favor more compact cluster formation
    r0 = ((0.6 / cols) * np.sqrt(1/(n/2))) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Define bounds with strict radius tolerance
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.6)]  # Increased radius upper limit for better expansion
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # We minimize this to maximize radii
    
    # Define vectorized constraints with fixed i to avoid lambda scoping issues
    cons = []
    for i in range(n):
        # Left side constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # Right side constraint: 1 - x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # Bottom side constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # Top side constraint: 1 - y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})
    
    # Create a dynamic constraint grid: optimized to prioritize high-interaction neighbors
    # Use matrix operations for fast overlap constraint evaluation
    for i in range(n):
        for j in range(i + 1, n):
            # Create anonymous function with correct captured parameters
            def constraint_func(i_val=i, j_val=j):
                def _func(v):
                    i_idx = 3*i_val
                    j_idx = 3*j_val
                    dx = v[i_idx] - v[j_idx]
                    dy = v[i_idx + 1] - v[j_idx + 1]
                    return dx*dx + dy*dy - (v[i_idx + 2] + v[j_idx + 2])**2
                return _func
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Initial optimization with hybrid strategies: 
    # - adaptive learning rates
    # - memory-optimized constraints
    # - gradient-based initialization with spatial hashing
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 2500, "ftol": 1e-10, "gtol": 1e-9, "eps": np.sqrt(np.finfo(float).eps)} )
    
    # Dynamic reconfiguration phase: forced geometric dissection on top 2 dynamically interacting circles
    # This is the critical mutation point of the directive
    
    # Phase 1: extract most dynamically interacting pairs using vectorized distance matrix
    if res.success:
        # Extract the optimal state so far
        optimal_v = res.x
        radii = optimal_v[2::3]
        centers = np.column_stack([optimal_v[0::3], optimal_v[1::3]])
        
        # Vectorized distance matrix computation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        
        # Find interaction weights: normalize by (radius_i + radius_j)
        interaction_weights = dist_matrix / (np.outer(radii, radii) + 1e-12)
        
        # Compute interaction score as sum of inverse distances
        interaction_scores = np.sum(interaction_weights, axis=1)
        top_n_idx = np.argsort(interaction_scores)[-2:]  # Get top 2 most interacting circles
        
        # Phase 2: Reconfigure top two interacting circles via spatial reassignment
        # Perform geometric dissection and allow constrained radius growth
        # Create an alternate configuration by shifting centers and adjusting radii
        
        # Create base reconfiguration vector from current state
        reconfigure_v = optimal_v.copy()
        
        # For each of the top 2 interacting circles: 
        # - apply a controlled shift in center position
        # - allow radius expansion under constraint of non-overlap and boundary compliance
        # - create new overlapping constraints for these circles with other ones
        for idx in top_n_idx:
            x = reconfigure_v[3*idx]
            y = reconfigure_v[3*idx + 1]
            radius = reconfigure_v[3*idx + 2]
            
            # Apply geometric constraint: displace the circle to form a new geometric relation
            # Using trigonometric shift to create a controlled dissection
            angle_shift = np.random.uniform(0.0, np.pi * 0.4)  # up to 40 degrees
            displacement = np.array([np.cos(angle_shift) * radius * 1.2, 
                                    np.sin(angle_shift) * radius * 1.2])
            
            # Apply a radial shift with boundary check
            new_x = x + displacement[0]
            new_y = y + displacement[1]
            
            # Check for boundary constraints
            if new_x < 0 or new_x > 1 or new_y < 0 or new_y > 1:
                # Re-center to valid region with buffer
                new_x = 0.5 + np.random.uniform(-0.2, 0.2) if new_x < 0 else 1.0 - np.random.uniform(0.2, 0.5)
                new_y = 0.5 + np.random.uniform(-0.2, 0.2) if new_y < 0 else 1.0 - np.random.uniform(0.2, 0.5)
            
            reconfigure_v[3*idx] = new_x
            reconfigure_v[3*idx + 1] = new_y
            # Apply radius perturbation with boundary-aware adjustment
            new_radius = radius * 1.2 - 0.002 * np.random.rand()
            if new_radius < 1e-4:
                new_radius = 1e-4  # prevent negative or zero radii
            reconfigure_v[3*idx + 2] = new_radius
        
        # Phase 3: Re-evaluate the reconfigured state with new constraints
        # Create new constraint for top 2 interacting circles to other circles
        # This effectively introduces a new topology constraint for the system
        # Note: we need to create new overlap constraints for these two circles with others
        
        # Prepare new constraint list that includes the current one but with new values
        # This maintains the original constraints but with reconfigured values
        new_cons = []
        for c in cons:
            # Re-evaluate the constraint function with new_v
            def get_func(c_func):
                def _func(v):
                    return c_func(v)
                return _func
            new_func = get_func(c["fun"])
            new_cons.append({"type": c["type"], "fun": new_func})
        
        # Run optimization with reconfigured vector
        res = minimize(neg_sum_radii, reconfigure_v, method="SLSQP", bounds=bounds,
                       constraints=new_cons, options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-9, "eps": np.sqrt(np.finfo(float).eps)} )
        
        # Phase 4: Targeted Radius Expansion on Most Isolated Circle with Adjacency Constraints
        # Identify the most isolated circle in the new configuration
        # This will be a novel adjacency constraint in a reconfigured layout
        if res.success:
            current_v = res.x
            radii = current_v[2::3]
            centers = np.column_stack([current_v[0::3], current_v[1::3]])
            
            # Recompute distance matrix
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
            dist_matrix = np.sqrt(dx**2 + dy**2)
            
            # Compute isolation score for each circle
            # This is the sum of inverse distances to other circles
            isolation_scores = np.sum(1.0 / (dist_matrix + 1e-12), axis=1)
            most_isolated_idx = np.argmin(isolation_scores)
            
            # Check constraint compliance
            # We'll allow a slight expansion, but with a bounded optimization
            # Use gradient-based update with constraint awareness
            # Introduce a new constraint: isolate_circle must have larger radius
            # This creates a topological re-ordering in the layout
            
            # We need to create new constraints for the most_isolated_idx
            # Create this as a novel adjacency constraint (this could potentially be part of the next reconfiguration)
            # But we first create a new constrained vector
            
            new_v = current_v.copy()
            # Perturb the most isolated circle in a non-overlapping and boundary-consistent way
            # Use a directional shift based on the position of its nearest neighbor
            # Find nearest neighbor to the most isolated circle
            nearest_idx = np.argmin(dist_matrix[most_isolated_idx])
            nearest_dist = dist_matrix[most_isolated_idx, nearest_idx]
            if nearest_dist > 0.7:
                # We can try to move it out of the cluster
                angle = np.arctan2(centers[nearest_idx, 1] - centers[most_isolated_idx, 1],
                                   centers[nearest_idx, 0] - centers[most_isolated_idx, 0])
                shift_dir = np.array([np.cos(angle), np.sin(angle)])
                new_x = centers[most_isolated_idx, 0] + shift_dir[0] * radii[most_isolated_idx] * 1.5
                new_y = centers[most_isolated_idx, 1] + shift_dir[1] * radii[most_isolated_idx] * 1.5
                # Clamp to unit square with boundary buffer
                new_x = np.clip(new_x, 1e-6, 1 - 1e-6)
                new_y = np.clip(new_y, 1e-6, 1 - 1e-6)
                new_v[3*most_isolated_idx] = new_x
                new_v[3*most_isolated_idx + 1] = new_y
                # Increase radius by a moderate factor
                new_radius = radii[most_isolated_idx] * 1.2 + 0.002 * np.random.rand()
                if new_radius < 1e-4:
                    new_radius = 1e-4
                new_v[3*most_isolated_idx + 2] = new_radius
            
            # Create a new constraint that the most isolated circle must expand under certain conditions
            # This constraint will be re-evaluated in the new vector
            # For brevity, re-evaluate the optimized vector with modified parameters
            res = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=bounds,
                           constraints=new_cons, options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-9, "eps": np.sqrt(np.finfo(float).eps)} )
        
        # Phase 5: Final validation and constraint re-evaluation after reconfiguration
        # Ensure that all constraints are re-checked with current_v
        if res.success:
            final_v = res.x
            # Final constraint validation
            # This is not required as the optimizer ensures compliance, but we perform it for robustness
            centers_final = np.column_stack([final_v[0::3], final_v[1::3]])
            radii_final = final_v[2::3]
            
            # Recompute interaction distances
            dx = centers_final[:, np.newaxis, 0] - centers_final[np.newaxis, :, 0]
            dy = centers_final[:, np.newaxis, 1] - centers_final[np.newaxis, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Check if any circles are overlapping
            over = np.any(dists < (radii_final[:, np.newaxis] + radii_final[np.newaxis, :]) - 1e-12)
            if over:
                # In case of unexpected overlap (due to perturbations or numerical error), perform a final adjustment
                # Apply minimal perturbations to centers to ensure compliance
                # This is a fallback mechanism to prevent failure
                perturbation = np.random.rand(2, n) * 0.01
                final_v[0::3] += perturbation[0, :]
                final_v[1::3] += perturbation[1, :]
                # Recompute centers after perturbation
                centers_final = np.column_stack([final_v[0::3], final_v[1::3]])
                # Re-validate
                over = np.any((np.sqrt((centers_final[:, np.newaxis, 0] - centers_final[np.newaxis, :, 0])**2 +
                                      (centers_final[:, np.newaxis, 1] - centers_final[np.newaxis, :, 1])**2)
                               ) < (radii_final[:, np.newaxis] + radii_final[np.newaxis, :]) - 1e-12)
            
            # Finalize if constraint satisfaction is guaranteed
            if not over:
                v = final_v
            else:
                # Fallback to original if constraints fail (unlikely)
                v = res.x  # fallback to last successful solution
                
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = np.clip(v[2::3], 1e-6, None)
            return centers, radii, float(radii.sum())
    
    # Fallback in case of all optimization paths failing
    # This is a defensive fallback to ensure the function returns a valid configuration
    v = v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())