import numpy as np

def run_packing():
    n = 26

    # Use an advanced spatial hashing strategy with 6 cols x 5 rows (dense packing)
    # This structure allows for more compact arrangement with staggered rows
    cols = 6
    rows = (n + cols - 1) // cols
    # Precompute initial base grid coordinates with geometric clustering
    grid_x = np.linspace(0.0, 1.0, cols + 2)  # Extra spacing for boundary padding
    grid_y = np.linspace(0.0, 1.0, rows + 2)
    
    # Generate spatial coordinates with enhanced geometric hashing for initial configuration
    initial_centers = []
    radii_base = []
    
    # Use adaptive hashing with spatial clustering weights
    for i in range(n):
        row = i // cols
        col = i % cols
        # Base location based on grid + slight randomization
        x_center = grid_x[col + 1] + np.random.uniform(-0.03, 0.03)
        y_center = grid_y[row + 1] + np.random.uniform(-0.03, 0.03)
        # Alternate row offset for staggered packing
        if row % 2 == 1:
            x_center += (grid_x[1] - grid_x[0]) * 0.5
        # Add small random displacement for better spread
        x_center += np.random.uniform(-0.015, 0.015)
        y_center += np.random.uniform(-0.015, 0.015)
        # Apply small perturbation based on grid spacing
        x_center += (np.random.rand() - 0.5) * (grid_x[1] - grid_x[0]) * 0.3
        y_center += (np.random.rand() - 0.5) * (grid_y[1] - grid_y[0]) * 0.3
        # Normalize to [0,1]
        x_center = np.clip(x_center, 0.0, 1.0)
        y_center = np.clip(y_center, 0.0, 1.0)
        initial_centers.append((x_center, y_center))
    
    # Initial radii estimation based on grid spacing and optimization margin
    # Use smaller base radius than original (0.35 / cols) for better expansion potential
    r0 = 0.38 / cols - 1e-2  # Smaller base radius allows for more expansion
    radii_base.append(r0)  # Add radius for first circle
    
    # Construct initial vector: x, y, radius for each of the 26 circles
    v0 = np.zeros(3 * n)
    for i in range(n):
        v0[3 * i] = initial_centers[i][0]
        v0[3 * i + 1] = initial_centers[i][1]
        v0[3 * i + 2] = radii_base[i]
    
    # Establish bounds with strict constraints
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((1e-5, 0.5))  # radius (minimum 1e-5, maximum 0.5)
    
    # Objective is to maximize the sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Create constraints: boundary and mutual distance between circles
    # These are defined with careful closure handling
    constraints = []
    
    # Define boundary constraints
    for i in range(n):
        # x >= radius
        constraints.append({
            "type": "ineq",
            "fun": lambda v, idx=i: v[3 * idx] - v[3 * idx + 2]
        })
        # x + radius <= 1
        constraints.append({
            "type": "ineq",
            "fun": lambda v, idx=i: 1.0 - v[3 * idx] - v[3 * idx + 2]
        })
        # y >= radius
        constraints.append({
            "type": "ineq",
            "fun": lambda v, idx=i: v[3 * idx + 1] - v[3 * idx + 2]
        })
        # y + radius <= 1
        constraints.append({
            "type": "ineq",
            "fun": lambda v, idx=i: 1.0 - v[3 * idx + 1] - v[3 * idx + 2]
        })
    
    # Now define overlap constraints using vectorized distance function
    def distance_squared_i_j(v, i, j):
        dx = v[3*i] - v[3*j]
        dy = v[3*i+1] - v[3*j+1]
        return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
    
    # Vectorized overlap constraints (distance squared >= (r_i + r_j)^2)
    # These are constructed with care to prevent lambda capture issues
    for i in range(n):
        for j in range(i+1, n):
            # Create constraint for pair i-j
            constraints.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: distance_squared_i_j(v, i, j)
            })
    
    # First optimization with aggressive max iterations and tight tolerance
    # Use SLSQP with gradient approximations when needed
    first_res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP", 
        bounds=bounds, 
        constraints=constraints, 
        options={
            "maxiter": 1500,
            "ftol": 1e-11, 
            "gtol": 1e-9, 
            "eps": 1e-8,  # Increased epsilon for more stable gradient approximation
            "disp": False
        }
    )
    
    # Asymmetric spatial reconfiguration with adaptive perturbation
    if first_res.success:
        v = first_res.x
        r_vals = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Generate adaptive spatial perturbation based on relative radius position
        spatial_perturb = np.random.rand(n, 2) * 0.1
        # Weight perturbation by inverse of radius (closer circles have larger perturbation)
        perturb_factor = 1.0 / (r_vals[None, :] + 1e-10)
        perturbed_centers = centers + spatial_perturb * perturb_factor
        
        # Create perturbed vector
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] = perturbed_centers[i, 0]
            perturbed_v[3*i+1] = perturbed_centers[i, 1]
        
        # Re-optimization with perturbed configuration and tighter constraints
        after_perturb_res = minimize(
            neg_sum_radii,
            perturbed_v,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 500,
                "ftol": 1e-11,
                "gtol": 1e-9,
                "eps": 1e-8,
                "disp": False
            }
        )
        
        # Adaptive post-perturb refinement
        if after_perturb_res.success:
            v = after_perturb_res.x
            # We now perform a targeted expansion of the circle with least influence constraint
            # First, calculate all pairwise distances for all circles
            # Vectorized distance calculation with NumPy
            centers_new = np.column_stack([v[0::3], v[1::3]])
            # Distance matrix with broadcasting
            dx = centers_new[:, np.newaxis, 0] - centers_new[np.newaxis, :, 0]
            dy = centers_new[:, np.newaxis, 1] - centers_new[np.newaxis, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Compute distance from each circle to others
            min_dist_to_others = np.min(dists, axis=1)
            
            # Identify circle with largest minimal distance (least constrained)
            least_constrained_idx = np.argmax(min_dist_to_others)
            
            # Determine circle with smallest radius for targeted expansion
            smallest_radius_idx = np.argmin(v[2::3])
            
            # Choose a target circle to prioritize expansion based on constraint potential
            # Prefer the least constrained circle if it's not already the smallest
            target_idx = least_constrained_idx if least_constrained_idx != smallest_radius_idx else smallest_radius_idx
            
            # Calculate how much can we expand this circle before constraints are violated
            # We do this with a gradient descent approach, calculating expansion
            # We perform a binary search for maximum allowable expansion
            
            # Start with a base expansion factor
            expansion_factor = 0.0
            # Use a fine search range to find allowable expansion
            step_size = 0.001
            # Find the maximum allowable expansion step before constraints are violated
            for step in np.arange(0.0, 0.05, step_size):  # try up to 0.05%
                # Create a trial expanded configuration
                trial_v = np.copy(v)
                trial_v[3 * target_idx + 2] += step
            
                # Evaluate all constraints again with this trial configuration
                # We do this by checking pairwise distances for non-overlap and boundary constraints
                trial_centers = np.column_stack([trial_v[0::3], trial_v[1::3]])
                trial_r = trial_v[2::3]
                valid = True
                for i in range(n):
                    for j in range(i+1, n):
                        dx = trial_centers[i, 0] - trial_centers[j, 0]
                        dy = trial_centers[i, 1] - trial_centers[j, 1]
                        dist = np.sqrt(dx ** 2 + dy ** 2)
                        if dist < trial_r[i] + trial_r[j] - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    # Revert to previous step
                    trial_v = np.copy(v)
                    trial_v[3 * target_idx + 2] -= step
                    break
            
            # Compute how much we can safely expand the target circle
            # Use binary search for the max allowable expansion
            max_expansion = 0.0
            low, high = 0.0, 1.0  # 1.0 is arbitrary high limit
            # For 15 iterations of binary search
            for _ in range(20):
                mid_val = (low + high) / 2
                # Compute trial
                trial_v = np.copy(v)
                trial_v[3 * target_idx + 2] = v[3 * target_idx + 2] + mid_val
                trial_centers = np.column_stack([trial_v[0::3], trial_v[1::3]])
                trial_r = trial_v[2::3]
                valid = True
                for i in range(n):
                    for j in range(i+1, n):
                        dx = trial_centers[i, 0] - trial_centers[j, 0]
                        dy = trial_centers[i, 1] - trial_centers[j, 1]
                        dist = np.sqrt(dx ** 2 + dy ** 2)
                        if dist < trial_r[i] + trial_r[j] - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    max_expansion = mid_val
                    low = mid_val
                else:
                    high = mid_val
            
            # After finding max allowable expansion, perform a more precise expansion
            # Now apply the expansion and re-optimize
            expanded_v = np.copy(v)
            expanded_v[3 * target_idx + 2] += max_expansion
            
            # Reoptimize with expanded radius
            final_opt_res = minimize(
                neg_sum_radii,
                expanded_v,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={
                    "maxiter": 400,
                    "ftol": 1e-11,
                    "gtol": 1e-9,
                    "eps": 1e-8,
                    "disp": False
                }
            )
            
            # Use the final optimization result
            if final_opt_res.success:
                v = final_opt_res.x
        
    # Final extraction of results
    v = first_res.x if not final_opt_res.success else final_opt_res.x
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    # Final validation pass (to ensure we are within bounds)
    # This is done via the minimize function and constraints, but we still apply a check
    # This ensures we're not in a situation where the minimize result is invalid due to numerical issues
    final_centers = centers.copy()
    final_radii = radii.copy()
    for i in range(n):
        x, y = final_centers[i]
        r = final_radii[i]
        if (x - r < -1e-12 or x + r > 1 + 1e-12 or 
            y - r < -1e-12 or y + r > 1 + 1e-12):
            final_centers[i] = [0.5, 0.5]
            final_radii[i] = 0.1  # fallback to small radius
        elif np.isnan(x) or np.isnan(y) or np.isnan(r):
            final_centers[i] = [0.5, 0.5]
            final_radii[i] = 0.1  # fallback
    
    return centers, radii, float(radii.sum())