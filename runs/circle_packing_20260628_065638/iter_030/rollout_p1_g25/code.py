import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with advanced geometric partitioning, dynamic spacing, and adaptive randomization
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        col_offset = 0.5 / cols
        row_offset = 0.5 / rows
        
        # Adaptive base coordinates for geometric balance
        x_center = (col + 0.5) * col_offset
        y_center = (row + 0.5) * row_offset
        
        # Use adaptive randomization based on spatial locality and row parity
        # Add more structured, non-linear spatial perturbation
        # Row-based non-uniform perturbation
        row_perturb = np.random.uniform(-0.1, 0.1) * (1 - row / rows)  # More randomness in higher rows
        col_perturb = np.random.uniform(-0.1, 0.1) * (1 - col / cols)  # More randomness in higher cols
        x = x_center + col_perturb
        y = y_center + row_perturb
        
        # Alternate row offset for staggered grid (more pronounced with rows)
        if row % 2 == 1:
            # Staggered shift but with smaller amplitude to maintain stability
            x += 0.4 / cols * (1 - row / rows)  # Adjusted magnitude based on placement
        
        xs.append(x)
        ys.append(y)
    
    r0 = 0.28 / cols - 1e-3  # More conservative initial radii to avoid collisions
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Construct strict bounds list with 3*n entries to match decision vector (length 3*n)
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    # Define the objective: negative sum of radii - minimize is equivalent to maximize
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Build constraints for boundary conditions (inqualities)
    # Use lambda capture with i to prevent closure ambiguity and lambda capture issues
    constraint_builder = lambda i: [
        {"type": "ineq", "fun": lambda v: v[3*i] - v[3*i+2]},  # x - r >= 0
        {"type": "ineq", "fun": lambda v: 1.0 - v[3*i] - v[3*i+2]},  # 1 - x - r >= 0
        {"type": "ineq", "fun": lambda v: v[3*i+1] - v[3*i+2]},  # y - r >= 0
        {"type": "ineq", "fun": lambda v: 1.0 - v[3*i+1] - v[3*i+2]}   # 1 - y - r >= 0
    ]
     # Use list comprehension for efficient construction of 4* n constraints
    cons = [constraint for i in range(n) for constraint in constraint_builder(i)]

    # Build constraints for circle-to-circle proximity
    # Use vectorized overlap constraints with proper i,j capture to avoid closure issues
    for i in range(n):
        for j in range(i + 1, n):
            # Use nested capture with i and j to avoid lambda closure issues
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                distance_squared = dx*dx + dy*dy
                total_radius = v[3*i+2] + v[3*j+2]
                # Ensure non-overlap by enforcing: distance^2 - (r_i + r_j)^2 >= 0
                return distance_squared - total_radius**2
            cons.append({"type": "ineq", "fun": constraint_func})
    
    # Initial optimization with refined hyperparameters to reduce divergence
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "eps": 1e-12})
    
    # Multi-stage reconfiguration with targeted spatial and radii reordering
    
    # Stage 1: Adaptive reconfiguration with spatial hashing and radius-aware perturbation
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Vectorize distances and compute constraints for all pairs
        # This is computationally expensive but ensures constraint validity
        # Instead of repeating during optimization, we perform this here in a reconfiguration stage
        # For robustness, we'll perform a secondary optimization with refined bounds
        # Spatial hashing with adaptive radius-based perturbation
        # Perturb spatial coordinates based on radius and spatial proximity
        # Create a grid of radii-aware spatial hash for reordering
        spatial_hash = np.random.rand(n, 2) * (0.05 * np.sqrt(1 + radii))
        perturbed_v = v.copy()
        for i in range(n):
            # Use radius as scaling factor for spatial perturbation
            radius_normalized = radii[i] / (r0 + 0.01)  # Avoid division by zero
            # Apply small spatial reconfiguration using radius-weighted perturbation
            perturbed_v[3*i] += spatial_hash[i, 0] * radius_normalized
            perturbed_v[3*i+1] += spatial_hash[i, 1] * radius_normalized
        
        # Re-evaluate
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-12})
    
    # Stage 2: Dynamic radius reordering with constrained growth
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute a spatial "constrained radius" metric
        # We identify the circle with the maximum minimal distance to neighbors to target for expansion
        # This ensures that we do not disturb circles with many neighbors
        dists = np.zeros((n, n))
        dx = np.expand_dims(centers[:, 0], axis=1) - centers[:, 0]
        dy = np.expand_dims(centers[:, 1], axis=1) - centers[:, 1]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = np.min(dists, axis=1)
        constrained_idx = np.argmax(min_dists)
        
        # Compute current sum and set a more conservative and realistic target to ensure feasibility
        current_total = np.sum(radii)
        # Estimate max potential based on grid and circle size
        # We assume maximum packing is around 2.7 (the current record is about 2.634)
        # But use more dynamic estimate based on current radii
        # Radius scaling can be improved by considering current average radius and spacing
        # Here we add an incremental 0.005 to current_total and use a scaling with average
        target_total = current_total + 0.005 * (1 + np.std(radii)/np.mean(radii))
        
        # To make growth feasible, instead of uniform expansion, we:
        # - Focus on the constrained circle
        # - Apply a radius growth that depends on spatial distance and minimal neighbor proximity
        # - Limit growth based on the minimal constraint (nearest neighbor distance)
        
        # We perform a two-phase optimization
        # First phase: reconfigure with radius expansion of constrained circle
        # Second phase: perform optimization under the new radius vector
        
        # Create a new_radii vector where we scale constrained_radius to reach target
        # But only if the change is feasible based on neighbor distances
        # If the expansion is not possible, we apply a constrained expansion
        # First, check how much we can expand the constrained circle
        
        # Compute the minimal expansion delta that doesn't cause overlap
        # This ensures we don't break constraints
        expanded_radii = radii.copy()
        max_growth = np.nan
        for j in range(n):
            if j == constrained_idx:
                continue
            dx = centers[j, 0] - centers[constrained_idx, 0]
            dy = centers[j, 1] - centers[constrained_idx, 1]
            dist = np.hypot(dx, dy)
            proposed_growth = (dist - radii[constrained_idx] - radii[j]) / 2.0
            if proposed_growth > 0:
                max_growth = max(max_growth, proposed_growth)
        
        # Now, compute potential delta that respects both current expansion target and constraints
        if np.isnan(max_growth):
            # No possible growth, no need to expand further
            new_growth = 0.0
        else:
            # Use a combination of target and constraints in a more efficient way
            # First compute how much we can grow the constrained_radius while others are held constant
            # This helps identify constrained and feasible space
            constrained_growth = (target_total - current_total) / (n) * 1.5  # Scaled up to ensure potential
            constrained_growth = min(constrained_growth, max_growth)  # Enforce constraint
            expanded_radii[constrained_idx] += constrained_growth
        
        # Re-evaluate configuration after growth with optimized constraints
        v = v.copy()
        v[2::3] = expanded_radii
        # Re-optimize
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-12})
    
    # Stage 3: Final refined optimization with constraint-based spatial reordering
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        # Compute again distance matrix for final validation
        dx = np.expand_dims(centers[:, 0], axis=1) - centers[:, 0]
        dy = np.expand_dims(centers[:, 1], axis=1) - centers[:, 1]
        dists = np.sqrt(dx**2 + dy**2)
        # Compute the minimal distance for each radius
        min_dists = np.min(dists, axis=1)
        # Identify the circle that is most "solitary" (most isolated)
        isolated_idx = np.argmax(min_dists)
        # Reorder circles by placing this isolated circle in the center, then pack others around
        # This is another heuristic to ensure robustness of the packing
        
        # Reorder the circles in a new order based on isolation to enhance symmetry and packing
        new_order = np.arange(n)
        new_order[[isolated_idx, 0]] = new_order[[0, isolated_idx]]
        
        # Perform a swap-based reordering of the decision vector
        # This ensures a different search region
        reordered_v = v.copy()
        for i in range(n):
            # Swap centers[i] and centers[new_order[i]]
            # This creates a new configuration with same circle sizes but different positions
            reordered_v[3*i], reordered_v[3*new_order[i]] = reordered_v[3*new_order[i]], reordered_v[3*i]
            reordered_v[3*i+1], reordered_v[3*new_order[i]+1] = reordered_v[3*new_order[i]+1], reordered_v[3*i+1]
        
        # Re-optimize after reordering
        res = minimize(neg_sum_radii, reordered_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-12})
    
    # Final check and cleanup
    v = res.x if res.success else v0
    # Enforce a hard clipping to prevent numerical errors
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)  # Ensure radii are positive and under max bound
    return centers, radii, float(radii.sum())