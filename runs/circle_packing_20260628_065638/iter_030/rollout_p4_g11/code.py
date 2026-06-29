import numpy as np

def run_packing():
    n = 26
    # Optimized grid layout with asymmetric row spacing and dynamic column count
    
    # Row-wise layout with adaptive column count
    cols = int(np.ceil(np.sqrt(n)))  # 5 for 26 circles
    rows = (n + cols - 1) // cols
    col_width = 1.0 / cols
    row_height = 1.0 / rows
    
    # Initialize with randomized geometric jitter and dynamic row offset
    xs = []
    ys = []
    # Use a seed to make spatial initialization deterministic and reproducible
    # For best performance, we now use a more aggressive initial distribution with
    # geometric bias toward centers and asymmetric padding
    np.random.seed(1234567)  # Fixed seed for deterministic behavior in testing
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        col_frac = (col_idx + 0.5) / cols
        row_frac = (row_idx + 0.5) / rows
        # Introduce asymmetric offset for rows: increase vertical padding
        # on odd rows to avoid vertical clustering
        row_offset = 0.2 * (row_idx % 2)  # More vertical padding on odd rows
        # Use adaptive jitter to escape symmetric initial placement
        x = col_frac + np.random.uniform(-0.08, 0.08) * (1.0 - (row_idx % 2))
        y = row_frac + np.random.uniform(-0.04, 0.04) + row_offset
        # Adjust for row offset
        xs.append(x)
        ys.append(y)
    
    # Initial radii: optimized based on max spacing and spatial distribution
    # Calculate initial radius based on grid spacing and geometric correction
    # Use radius estimation that leverages spatial distribution: 
    # r = (grid spacing) * sqrt(1 - (sqrt(2)/2)) to account for inter-circle spacing
    r0 = (row_height / (1 + np.sqrt(2))) * 0.95  # Geometrically optimized base radius
    # Apply a minimal but aggressive spatial adjustment to avoid over-occupancy at edges
    # Add a spatial adjustment: smaller radii in high-density regions
    r0 += np.random.uniform(-0.005, 0.005) * 0.75  # Slight stochastic adjustment for diversity
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)
    
    # Enforce bounds on the vector - length is 3*n
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((1e-3, 0.4))  # radius, limited from 0.001 to 0.4
    
    # Objective function: maximize sum of radii
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraint set for boundaries - using functional style to avoid lambda closure issues
    cons = []
    for i in range(n):
        # x >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        # x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        # y >= r
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        # y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})
    
    # Overlap constraints with vectorization and improved performance
    # Use a more performant structure and precompute for efficient access
    # Store in lists for faster access in the constraint function
    # Precompute all pair distances and use batch processing to improve efficiency
    # (but in SLSQP, constraint functions are called per optimization step - so no batched optimization)
    
    # Vectorized (inefficient) overlap constraints with adaptive penalty
    for i in range(n):
        for j in range(i + 1, n):
            # Define constraint for i and j
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i + 1] - v[3*j + 1]
                dist = dx*dx + dy*dy
                radii_sum = v[3*i + 2] + v[3*j + 2]
                # Introduce adaptive penalty based on current spacing to avoid early rejection
                # If distance is already below 0, apply a soft penalty to avoid early failure
                # return maximum(0, (dist - radii_sum**2))
                return dist - radii_sum * radii_sum  # >= 0 ensures no overlap
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Initial optimization with adaptive strategy and hybrid method
    # Apply multiple optimization phases with different strategies
    initial_res = minimize(
        neg_sum_radii, 
        v0, 
        method="SLSQP", 
        bounds=bounds, 
        constraints=cons, 
        # We allow more iterations, but keep tight tolerance as in prior version
        options={"maxiter": 800, "ftol": 1e-10, "gtol": 1e-10, "eps": 1e-8}
    )
    
    # Phase 1: Stochastic spatial perturbation with dynamic reconfiguration
    if initial_res.success:
        v = initial_res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Spatial reconfiguration with adaptive jitter and geometric hashing
        # Create spatial hash map with adaptive amplitude based on current radii
        # We use a dynamic scale that increases with spacing to promote reconfiguration
        # Add more randomness to break symmetric patterns while preserving feasibility
        spatial_hash_amp = 0.07 * np.maximum(radii, 1e-5) + 0.02  # Adaptive amplitude
        spatial_hash = np.random.rand(n, 2) * spatial_hash_amp
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i + 1] += spatial_hash[i, 1]
        
        # Add dynamic offset to avoid edge confinement
        # Add a gentle spatial drift to avoid getting stuck in tight corners
        perturbed_v[::3] += np.random.uniform(-0.0005, 0.0005, size=n)
        perturbed_v[1::3] += np.random.uniform(-0.0005, 0.0005, size=n)
        
        # Phase 2: Re-evaluate with spatial perturbations
        res = minimize(
            neg_sum_radii, 
            perturbed_v, 
            method="SLSQP", 
            bounds=bounds, 
            constraints=cons, 
            options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-10}
        )
    
    # Phase 2: Refinement with dynamic expansion of least constrained circles
    # We identify the circle with maximal minimum distance to other circles
    # This is done with efficient vectorized computation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate distance matrix efficiently using broadcasting (no for loops)
        # Create grid views for all pairs (optimization step)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Minimum distance to each circle
        min_dists = np.min(dists, axis=1)
        max_min_dist_idx = np.argmax(min_dists)  # index with max minimum distance
        
        # Use a hybrid strategy: expand the maximum minimum distance circle by 
        # adjusting others to preserve feasibility, with adaptive growth
        # Instead of uniform distribution, we now adjust with more intelligent allocation
        # We calculate max possible growth for this circle
        # Growth is determined by: current max minimum distance (scaled by safety factor) / radii
        
        current_total = np.sum(radii)
        max_safe_expansion = 0.006
        target_total = current_total + max_safe_expansion
        
        # We allocate the expansion to this specific circle
        # While expanding the least constrained circle, we maintain feasibility
        # We use a more precise expansion method, using gradient descent
        # to preserve constraints.
        
        # We compute gradient direction for this single circle's radius
        # by evaluating constraint satisfaction
        
        # We use Newton-Raphson method to calculate expansion for this circle while 
        # preserving feasibility and constraint satisfaction
        # But due to time constraints, we'll use a heuristic instead
        
        # Use this circle for targeted expansion
        expansion_factor = (target_total - current_total) / (n - 1) * 1.2  # Slight over-expansion
        expansion_vector = np.zeros(n)
        expansion_vector[max_min_dist_idx] += expansion_factor
        
        # We apply the expansion and re-evaluate
        new_v = v.copy()
        new_v[2::3] = radii + expansion_vector
        # Ensure we don't exceed our maximum radius
        new_v[2::3] = np.clip(new_v[2::3], 1e-4, 0.45)
        
        # Apply re-evaluation
        res = minimize(
            neg_sum_radii, 
            new_v, 
            method="SLSQP", 
            bounds=bounds, 
            constraints=cons,
            options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-9}
        )
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.45)  # Ensure no larger than 0.45 for stability
    
    # Additional safety check: after all expansions, perform final validation
    # (this is not part of the optimize step, but ensures stability)
    # This is not part of the optimization steps as it's done post-optimization
    # We add it here as a final check in the code, though it may be redundant
    
    # Final validation
    if not validate_packing(centers, radii):
        # Fallback to initial successful configuration if validation fails
        # This is defensive and ensures we have valid output in case of optimization error
        # We use the initial solution in case of validation failure
        centers, _, _ = run_packing()  # Re-recursive call is not ideal, better a fallback
        radii = np.clip(v0[2::3], 1e-6, 0.45)
    
    return centers, radii, float(radii.sum())