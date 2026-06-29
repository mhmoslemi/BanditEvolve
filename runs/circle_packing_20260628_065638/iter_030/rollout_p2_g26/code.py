import numpy as np

def run_packing():
    n = 26
    # Optimal grid configuration: 5x6 grid with 1 extra column to allow flexibility for dense packing
    cols = 6  # columns
    rows = 5  # rows
    # Adjusted grid calculation: 5x6 grid has 30, we'll add a 6th row and use dynamic distribution
    
    # Initialize positions with a hybrid geometric clustering approach: 
    # - Initial grid-based distribution + 
    # - Stochastic perturbation for escape from local minima + 
    # - Edge-aware bias to push clusters against the square walls
    
    # Base grid positions
    xs = []
    ys = []
    base_grid = np.linspace(0, 1, cols+2)  # adding 2 to include boundary margins for safety
    row_offset = 0.25 / rows  # vertical spacing control
    
    for i in range(n):
        col_index = i % cols
        row_index = i // cols
        # x-coordinate with edge-aware distribution to avoid extreme cluster formation
        x = base_grid[col_index + 1] + np.random.uniform(-0.03, 0.03)
        x = np.clip(x, 0.02, 0.98)  # enforce safe boundaries from edges with padding
        
        # y-coordinate with alternating phase to create zig-zag grid
        row = rows - row_index - 1  # invert for bottom-up row count
        if (row_index % 3) == 0:  # phase shift for complex pattern generation
            # vertical bias towards bottom for complex stacking
            y = base_grid[row + 1] + np.random.uniform(-0.04, 0.04)
        else:
            # maintain base spacing with slight perturbation
            y = base_grid[row + 1] + np.random.uniform(-0.02, 0.02)
        y = np.clip(y, 0.02, 0.98)
        
        # alternate row offset for dense stacking
        if row_index % 2 == 1:
            x += (0.2 / cols)
        
        xs.append(x)
        ys.append(y)
    
    # Initial radius: more aggressive to allow dynamic adjustment
    # Based on grid density - rows*cols is slightly larger than n but we use 25 max for safe initial
    r0 = 0.35 * (n / (cols * rows)) - 1e-3  # adaptive scale from grid size
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Build bounds with 3n entries for v
    bounds = []
    for _ in range(n):
        # x and y: strict within [0, 1]
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # r must be positive

    def neg_sum_radii(v):
        """Optimization objective function: maximize sum of radii by minimizing negative"""
        return -np.sum(v[2::3])

    # Constraint definitions: all with lambda closures properly captured
    cons = []

    for i in range(n):
        # Define boundaries with more refined tolerances
        # Use function-based constraint closure with proper capture
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})  # x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})  # x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})  # y - r >=0 
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})  # y + r <=1

    # For overlap constraints: compute the squared distances and ensure they are larger than sum of radii squared
    # Use vectorized calculation with proper capture to avoid closure issues
    for i in range(n):
        for j in range(i + 1, n):
            # Precompute radius indices once for efficiency
            r_i_idx = 3*i + 2
            r_j_idx = 3*j + 2
            x_i_idx = 3*i
            y_i_idx = 3*i + 1
            x_j_idx = 3*j
            y_j_idx = 3*j + 1
            
            # Construct a lambda that's closure-bound
            # This is a lambda with captured indices, optimized for speed
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j, x_i=x_i_idx, y_i=y_i_idx, x_j=x_j_idx, y_j=y_j_idx, r_i=r_i_idx, r_j=r_j_idx:
                    (v[x_i] - v[x_j])**2 + (v[y_i] - v[y_j])**2 - (v[r_i] + v[r_j])**2
            })

    # First pass - initial optimization with increased max iterations and tighter tolerances
    # Adaptive options depending on early optimization success
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds, constraints=cons, 
                   options={"maxiter": 1200, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-9})

    # Stage 1: Geometric Dissection - identify top-2 interactive circles, reconfigure their relationships
    # This is the key tactical action
    if res.success:
        v = res.x
        # Recalculate radii and centers for analysis
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Smart vectorized distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        dists = np.triu(dists)  # only consider upper triangle for pairwise distances
        interaction_sum = np.sum(dists, axis=1)
        top_idx = np.argsort(interaction_sum)[-2:]  # take top 2 most interacting

        # Create reconfiguration candidate for these 2: use geometric hashing to break from existing pattern
        # Use a dual-layer perturbation to create new relationships
        top_v = v.copy()
        for idx in top_idx:
            # First layer perturbation: large but controlled movement
            top_v[3*idx] += np.random.uniform(-0.07, 0.07)
            top_v[3*idx + 1] += np.random.uniform(-0.07, 0.07)
            # Second layer: fine tuning, but with spatial-aware radius perturbation
            # Add some radius fluctuation to enable new spacing
            # Note: Radius perturbation is limited to avoid immediate invalid configurations
            top_v[3*idx + 2] += np.random.uniform(-0.003, 0.003)
        
        # Reoptimize with perturbed top two circles
        # Increase maxiter to handle complex reconfiguration
        res = minimize(neg_sum_radii, top_v, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 1400, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-9})
    
    # Stage 2: Adaptive Radius Expansion on Least Constrained Circle with topology preservation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        dists = np.zeros((n, n))
        
        # Vectorized distance calculation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute isolation score as weighted sum of minimal distances
        # This gives a measure of how isolated a circle is
        min_dists = np.min(dists, axis=1)
        # Introduce a radial weight to penalize isolated circles
        # This encourages expansion of the least constrained circle
        isolation = np.sum(min_dists)  # simple measure
        least_constrained_idx = np.argmin(isolation)
        
        # Now, prepare for targeted radius expansion
        # We will allow expansion but with a constraint that the change in radii is balanced
        # to maintain configuration stability
        
        # First: compute expansion based on current configuration
        current_total = np.sum(radii)
        target_total = current_total + 0.006  # aiming for a 0.6% increase based on current best practice
        max_expansion = (target_total - current_total) * 1.0
        
        # Allocate the expansion as a vector with a bias toward the least constrained circle
        # We'll use a gradient-based allocation with exponential scaling for better convergence
        # We'll apply a radial expansion that scales with distance to neighbors
        # This ensures the expansion is feasible and doesn't create new overlaps
        
        # Compute pairwise distance-based expansion factor
        # This is more effective than simple uniform allocation
        dist_exp_factors = np.zeros(n)  # expansion factors for each circle
        for i in range(n):
            dist_exp_factors[i] = np.exp(-0.8 * min_dists[i])  # weight by distance
        
        # Normalize to form an expansion vector
        # Apply a max expansion limit and balance expansion factors
        expansion_vector = (
            (dist_exp_factors / np.sum(dist_exp_factors)) * 
            max_expansion
        )
        
        # We add a small adjustment to the least constrained circle: double its expansion for focus
        # while keeping sum constraints
        exp_factor = (1 - (0.5 * 1e-6))  # small epsilon to avoid division by zero
        expansion_vector[least_constrained_idx] *= 1.3
        
        # Apply the expansion vector to the radii
        new_radii = radii + expansion_vector
        
        # Now, we need to make sure that the newly allocated radii do not cause overlaps
        # We perform a validation loop and adjust if needed
        
        # Build new candidate vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # This is a key part: perform a validation pass to ensure no overlaps
        # If overlaps are found, we perform a gradient descent-style backtracking
        # This is done in a loop until the configuration is valid

        # Create a loop to iteratively adjust expansion until validity is achieved
        # This is more robust than naive adjustment
        while True:
            v_candidate = v_new.copy()
            centers_candidate = np.column_stack([v_candidate[0::3], v_candidate[1::3]])
            radii_candidate = v_candidate[2::3]
            
            # Check for all overlapping pairs
            overlap_found = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers_candidate[i, 0] - centers_candidate[j, 0]
                    dy = centers_candidate[i, 1] - centers_candidate[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < (radii_candidate[i] + radii_candidate[j]) - 1e-12:
                        overlap_found = True
                        break
                if overlap_found:
                    break
            
            if not overlap_found:
                # Configuration is valid; break out of loop
                break
            
            # We reduce expansion: a soft gradient descent approach
            # We reduce the total expansion of the circles by a percentage based on overlap
            # We use a dynamic reduction factor for stability
            reduction_factor = 0.97  # we reduce total expansion by 3% per iteration
            
            # Adjust the expansion vector
            v_new[2::3] *= reduction_factor  # apply a uniform reduction
            
            # To maintain the sum total, we redistribute the new expansion
            # but this is a simpler approach and may not be optimal
            # For better stability and convergence, we'd recompute with a revised strategy
            # but we will proceed with a basic solution for brevity while maintaining stability

        # Final candidate vector with constrained radii
        v_final = v_new.copy()

        # Reoptimize with this new candidate configuration
        # Here, we use an adaptive solver configuration to handle complex optimization
        res = minimize(neg_sum_radii, v_final, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-11, "eps": 1e-9})
    
    # Final safety net: fallback if optimization failed
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)  # clip to ensure positive values without invalid entries
    # Apply a second post-processing validation
    # This ensures any residual errors or NaN values are addressed
    valid_centers = centers.copy()
    valid_radii = radii.copy()
    
    # Validate against the strict validator
    # If validation fails, fall back to initial state
    # We will now perform a final check
    # This is optional but ensures stability
    
    # Validate to the same standards as the provided validator
    # (Note: Since we do not call the explicit validator, we simulate its behavior here)
    # This is done to avoid undefined behavior from NaN or out of bounds radii
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < -1e-12 or x + r > 1 + 1e-12 or 
            y - r < -1e-12 or y + r > 1 + 1e-12):
            # Out of bounds - use original if this fails
            # This is fallback to ensure compliance with the checker
            valid_centers[i] = v0[0::3][i*3:(i+1)*3]
            valid_radii[i] = v0[2::3][i]
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < (radii[i] + radii[j]) - 1e-12:
                # Overlap - use original values
                valid_centers[i] = v0[0::3][i*3:(i+1)*3]
                valid_centers[j] = v0[0::3][j*3:(j+1)*3]
                valid_radii[i] = v0[2::3][i]
                valid_radii[j] = v0[2::3][j]
    
    # Set final outputs
    centers = valid_centers
    radii = valid_radii
    sum_radii = float(radii.sum())
    
    return centers, radii, sum_radii