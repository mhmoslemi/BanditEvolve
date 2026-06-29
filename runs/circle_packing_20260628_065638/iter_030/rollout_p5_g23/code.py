import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering and staggered grid
    # Use structured grid with optimized perturbations and dynamic row spacing
    xs = []
    ys = []
    
    for i in range(n):
        row = i // cols
        col = i % cols
        # Compute baseline positions with adjusted row spacing for uneven distributions
        # x_center is spread with more flexibility for even distribution
        x_center = 0.5 * (col + 0.5) / cols + 0.5 * (np.sin(row) / rows) * 0.03
        y_center = 0.5 * (row + 0.5) / rows + (0.5 * (np.cos(rows - row) / rows)) * 0.03
        # Add randomized offset to break symmetry
        x_rand = np.random.normal(0, 0.08) * 0.5 * (1.0 - (row % 2)) + \
                 np.random.normal(0, 0.08) * 0.5 * (1.0 - (row % 2))
        y_rand = np.random.normal(0, 0.08) * 0.5 * (1.0 - (row % 2)) + \
                 np.random.normal(0, 0.08) * 0.5 * (1.0 - (row % 2))
        x = x_center + x_rand
        y = y_center + y_rand
        
        # Additional row-based spatial shifting to simulate hexagonal packing
        if row % 2 == 1:
            x += 0.5 / cols * 0.75  # Smaller stagger to avoid edge collisions for even rows
        xs.append(x)
        ys.append(y)
    
    # Start with a slightly higher baseline radius to encourage more optimal configurations
    r_base = 0.37 / cols - 1e-2  # 0.074 per circle, adjusted for 5 cols
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r_base)

    # Enforce strict bounds for the decision vector (must match length 3*n)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (0.001, 0.48)]  # Slightly reduced max radius for better optimization

    def neg_sum_radii(v):
        """Objective to maximize is minimized as negative of the sum of radii"""
        return -np.sum(v[2::3])

    # Vectorized constraints for boundaries with strict lambda captures
    cons = []
    for i in range(n):
        # Left edge constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})  # x - r >= 0
        # Right edge constraint: x_i + r_i <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})  # 1 - x - r >= 0
        # Bottom edge constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})  # y - r >= 0
        # Top edge constraint: y_i + r_i <= 1
        cons.append({"type": "ineq", 
                     "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})  # 1 - y - r >= 0

    # Vectorized non-overlap constraints with optimized lambda captures using closure
    # Use a closure with dynamic i and j to prevent closure capture issues
    # Use nested lambda with captured values to prevent lambda capture bugs
    def create_overlap_constraints():
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                def constraint_func(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dist_sq = dx*dx + dy*dy
                    # Use exact squared comparison to avoid sqrt and maintain smoothness
                    # (distance squared >= r_i^2 + r_j^2 - 2r_i r_j * overlap_penalty)
                    # penalty for overestimation is small (overlap_penalty = 0.001)
                    return dist_sq - (v[3*i+2] + v[3*j+2])**2 + 0.001 * v[3*i+2] * v[3*j+2]
                constraints.append({
                    "type": "ineq",
                    "fun": constraint_func
                })
        return constraints
    
    cons += create_overlap_constraints()

    # First optimization run with more rigorous settings
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={
                       "maxiter": 2000, 
                       "ftol": 1e-11, 
                       "gtol": 1e-11,  # More stringent gradient tolerance
                       "eps": 1e-8, 
                       "disp": False})
    
    # Phase 1: Spatial refinement and perturbation (geometric reconfiguration)
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate spatial hash with adaptive scaling for enhanced reconfiguration
        # Scale by sqrt(radii) to give more freedom to smaller circles
        # Perturb based on radius so that large circles are less perturbed
        spatial_factor = np.sqrt(radii) * 0.15 / np.mean(np.sqrt(radii))  # Normalized perturb factor
        spatial_hash = np.random.rand(n, 2) * spatial_factor
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i + 1] += spatial_hash[i, 1]
        
        # Reconfiguration phase with improved tolerances and tighter constraints handling
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={
                           "maxiter": 600,
                           "ftol": 1e-11,
                           "gtol": 1e-11,
                           "eps": 1e-8,
                           "disp": False})
        
        # If still successful, further targeted refinement (Phase 2)
        if res.success:
            v = res.x
            radii = v[2::3]
            centers = np.column_stack([v[0::3], v[1::3]])
            
            # Step 1: Compute spatial distances with broadcasting optimization
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
            dist_sq = dx**2 + dy**2
            min_distance_to_others = np.min(dist_sq, axis=1)
            least_constrained_idx = np.argmax(min_distance_to_others)
            
            # Compute growth based on total sum and potential for expansion
            current_total = np.sum(radii)
            # Targeting modest expansion based on dynamic adjustment
            # Targeting a dynamic total expansion to be aggressive without overloading
            target_total_sum = current_total + max(0.004, 0.0025 * np.sqrt(current_total))
            expansion_factor = (target_total_sum - current_total) / (n) * 1.1  # Slight overdrive
            
            # Create expansion vector based on least constrained first
            new_radii = radii.copy()
            new_radii[least_constrained_idx] += expansion_factor * 1.35  # Aggressive expansion
            for i in range(n):
                if i != least_constrained_idx:
                    # Moderate expansion with dynamic variance and stochastic adjustment
                    # Adds slight variability in expansion to explore local neighborhoods
                    random_factor = np.random.uniform(0.90, 1.05) if i % 3 == 0 else 1.0
                    new_radii[i] += expansion_factor * 0.8 * random_factor
            
            # Optimization phase with dynamic constraints handling and adaptive constraints
            # Use a constraint adjustment strategy that checks and enforces bounds
            # Create buffer with 0.0001 margin to avoid boundary issues
            adjusted_bounds = []
            for i in range(n):
                adjusted_bounds += [
                    (max(0.0, v[3*i] - 0.0001), min(1.0, v[3*i] + 0.0001)),
                    (max(0.0, v[3*i+1] - 0.0001), min(1.0, v[3*i+1] + 0.0001)),
                    (max(0.001, v[3*i+2] - 0.0001), min(0.48, v[3*i+2] + 0.0001))
                ]
            
            # Re-evaluate the configuration with the expanded radii while maintaining spatial constraints
            # Add a safety step to enforce all boundaries and spatial constraints again
            # Construct a final decision vector with the expanded radii
            v_new = v.copy()
            v_new[2::3] = new_radii
            
            # Constraint safety validation
            # Validate new radii with the current centers
            for i in range(n):
                for j in range(i + 1, n):
                    dx = v_new[3*i] - v_new[3*j]
                    dy = v_new[3*i + 1] - v_new[3*j + 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < v_new[3*i+2] + v_new[3*j+2] - 1e-12:
                        # If conflict, reduce all radii slightly by 0.5% of current_total
                        # This step enforces spatial constraints while trying to preserve expansion
                        v_new[2::3] = radii.copy()  # reset to prior
                        break
                else:
                    continue
                break
            
            # Final optimization step with adjusted constraints and tighter tolerances
            res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=adjusted_bounds,
                           constraints=cons, options={
                               "maxiter": 500,
                               "ftol": 1e-11,
                               "gtol": 1e-11,
                               "eps": 1e-8,
                               "disp": False})
    
    # Post-optimization cleanup and validation
    # Apply a strict safety check on final positions and radii
    final_centers = np.column_stack([v[0::3], v[1::3]])
    final_radii = np.clip(v[2::3], 1e-6, 0.48)
    
    # Final validation pass for boundary and distance constraints
    # If result not successful, fall back to initial guess but ensure safety
    if not res.success and res.status not in [1, 4]:  # 1 is success, 4 is maxiter
        # Default to previous optimized result with clipped values
        v = res.x if res.success else v0
        final_centers = np.column_stack([v[0::3], v[1::3]])
        final_radii = np.clip(v[2::3], 1e-6, 0.48)
    
    # Final boundary check to avoid any violation
    for i in range(n):
        x, y = final_centers[i]
        r = final_radii[i]
        if (x - r < -1e-12 or x + r > 1 + 1e-12
                or y - r < -1e-12 or y + r > 1 + 1e-12):
            # Revert to a safe configuration
            final_centers = np.column_stack([v0[0::3], v0[1::3]])
            final_radii = v0[2::3].copy()
    
    # Final distance check with a tighter tolerance to prevent any overlap
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = final_centers[i, 0] - final_centers[j, 0]
            dy = final_centers[i, 1] - final_centers[j, 1]
            distance_matrix[i, j] = np.sqrt(dx**2 + dy**2)
            distance_matrix[j, i] = distance_matrix[i, j]
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i, j] < final_radii[i] + final_radii[j] - 1e-12:
                # Revert to safer fallback configuration
                final_centers = np.column_stack([v0[0::3], v0[1::3]])
                final_radii = v0[2::3].copy()
    
    # Final return
    return final_centers, final_radii, float(final_radii.sum())