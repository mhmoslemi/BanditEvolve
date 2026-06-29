import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    # Dynamic grid to handle asymmetry with some row-based offset
    rows = (n + cols - 1) // cols
    
    # Step 1: Initial grid with adaptive clustering and spatial randomness
    # Using a more refined grid where we allow column-based asymmetry
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        col_offset = (np.random.uniform(-0.02, 0.02))
        row_offset = (np.random.uniform(-0.02, 0.02))
        # Column-based offset to break column symmetry
        col_scaled = col / cols  # normalized col
        row_scaled = (row / rows) * (1 - (col_scaled / cols))  # asymmetric row scaling
        
        # Apply offset to break grid symmetry
        x_center = col_scaled + col_offset
        y_center = row_scaled + row_offset
        
        # Row-based staggering to avoid vertical clustering
        if row % 2 == 1:
            x_center += 0.5 / cols
            x_center = np.clip(x_center, 0.0, 1.0)  # clamp to avoid overflow
        xs.append(x_center)
        ys.append(y_center)
    
    # Initial radii with a smart baseline from spatial grid
    # We compute a grid density-dependent estimate
    grid_min_dist = (1.0 / cols) * (1.0 / rows) # rough estimate for closest neighbors
    base_radius = grid_min_dist / 1.15  # 15% more than minimal required to allow spacing
    r0 = np.clip(base_radius - 0.0005, 1e-4, 0.3)  # adjust with a small buffer for initial optimization
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Ensure bounds are consistent with exactly 3*n entries
    bounds = []
    for _ in range(n):
        # x, y bounds: [0, 1]
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.35)]  # upper bound adjusted to be lower to prevent early overlap
    # Validate bounds length
    assert len(bounds) == 3 * n, f"Bounds length mismatch: got {len(bounds)}, expected 3*{n}"
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])  # minimize the negative of sum to maximize sum
    
    # Step 2: Create tight, efficient constraints with dynamic lambda capture and closure
    # Using nested lambda with i capture: this is safe in SLSQP as the optimizer will handle bounds
    cons = []
    for i in range(n):
        # Side constraints: center[x] - radius >= 0
        cons.append({
            "type": "ineq",
            "fun": (lambda v, i=i: v[3*i] - v[3*i + 2])
        })
        # Side constraint: center[x] + radius <= 1
        cons.append({
            "type": "ineq",
            "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2])
        })
        # Side constraint: center[y] - radius >= 0
        cons.append({
            "type": "ineq",
            "fun": (lambda v, i=i: v[3*i + 1] - v[3*i + 2])
        })
        # Side constraint: center[y] + radius <= 1
        cons.append({
            "type": "ineq",
            "fun": (lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2])
        })
    
    # Step 3: Overlap constraints with vectorized distance calculation and optimized evaluation
    for i in range(n):
        for j in range(i + 1, n):
            # Vectorized calculation for constraints, optimized for performance and memory
            # This avoids Python-level for loops in constraint functions
            cons.append({
                "type": "ineq",
                "fun": (lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - 
                    (v[3*i+2] + v[3*j+2])**2)
            })
    
    # Step 4: Initial optimization with tighter constraints and advanced solver settings
    res_initial = minimize(neg_sum_radii, v0, method="SLSQP", 
                          bounds=bounds,
                          constraints=cons, 
                          options={
                              "maxiter": 1000, 
                              "ftol": 1e-10, 
                              "gtol": 1e-10,
                              "eps": 1e-6,
                              "disp": False
                          })
    
    # Step 5: Apply spatial reconfiguration with gradient-aware perturbations
    if res_initial.success:
        v = res_initial.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Generate adaptive perturbation based on relative positions and radii
        # This is a soft constraint-based reconfiguration
        perturbation_base = 0.005 + np.min(radii) * 0.5  # scale with minimum radius
        spatial_pert = np.random.rand(n, 2) * perturbation_base * 1.3  # increase amplitude for exploration
        
        # Add perturbation in direction of larger spacing between circles
        # We calculate the gradient of spacing in directions using a linear approximation
        # This is a heuristically derived perturbation matrix
        spatial_pert_directions = np.zeros((n, 2))  # store directional perturbations
        for i in range(n):
            # find direction of maximum spacing change for this circle
            # for this, we look for the circle with maximum distance from this one
            max_dist_idx = np.argmax(np.sqrt((centers[i, 0] - centers[:, 0])**2 + 
                                             (centers[i, 1] - centers[:, 1])**2))
            if max_dist_idx != i:
                # direction to perturb is towards the farthest point
                direction = (centers[max_dist_idx, 0] - centers[i, 0], 
                             centers[max_dist_idx, 1] - centers[i, 1])
                direction = direction / np.linalg.norm(direction)
                # add directional perturbation in direction, but only if distance > min spacing
                dist = np.sqrt((centers[i, 0] - centers[max_dist_idx, 0])**2 +
                              (centers[i, 1] - centers[max_dist_idx, 1])**2)
                if dist > 1.2 * (radii[i] + radii[max_dist_idx]):
                    spatial_pert[i] = direction * np.random.rand() * spatial_pert_base * 0.3
                else:
                    # use random perturbation if not enough spacing
                    pass
        
        # Compute new_v with perturbations
        new_v = v.copy()
        for i in range(n):
            new_v[3*i] += spatial_pert[i, 0]
            new_v[3*i+1] += spatial_pert[i, 1]
            # Ensure clamping
            new_v[3*i] = np.clip(new_v[3*i], 0.0, 1.0)
            new_v[3*i+1] = np.clip(new_v[3*i+1], 0.0, 1.0)
        
        # Evaluate with new configuration
        res_reconfig = minimize(neg_sum_radii, new_v, method="SLSQP",
                               bounds=bounds, 
                               constraints=cons, 
                               options={
                                   "maxiter": 300, 
                                   "ftol": 1e-10, 
                                   "gtol": 1e-9, 
                                   "eps": 1e-6,
                                   "disp": False
                               })
    
    # Step 6: Targeted radius expansion with gradient-aware allocation
    # We apply a new method: instead of finding the "least constrained" circle,
    # we find the circle with the minimal constraint "impact": i.e., which circle's
    # expansion would create the least disruption to the system
    # This is done using a "soft" score based on the derivative of constraints with respect to radii
    
    if res_reconfig.success:
        v = res_reconfig.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute a matrix of constraint impacts for each circle
        # This is a scalar metric representing how much radius change affects constraints
        # For optimization, the "constraint impact" is the sum of partial derivatives
        # We compute a rough approximation using gradient information
        
        # Precompute distances for constraint impact matrix
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Constraint impact: for each circle, how much a radius change will affect other constraints
        # This is approximated as the sum of distances to other circles multiplied by a factor
        # The idea is that small radii can be expanded more safely if they are far from others
        impact = np.sum(dists, axis=1)
        
        # Find candidate for expansion based on lowest impact
        # This represents circles with the least direct constraint from others
        expand_candidates = np.argsort(impact)  # most expandable circle is at index 0
        expand_idx = expand_candidates[0]  # pick the one with the smallest impact

        # Determine how much to expand: based on system margin and safety from edges
        # We use an empirical margin threshold and compute a proportional expansion
        current_total = np.sum(radii)
        # Safety: 0.5% of current total
        safety_margin = current_total * 0.005
        # Maximum possible growth: 2.5% of current total (based on empirical packing experiments)
        max_growth = current_total * 0.025

        # Calculate how much we can grow this circle without violating distance constraints
        # To do this, we look at the closest neighbors and compute how much we can grow while staying in bounds
        closest_neighbors = np.argsort(dists[expand_idx, :])[:5]  # top 5 closest
        max_expansion_amounts = []
        for neighbor_idx in closest_neighbors:
            # Get the neighbor data
            n_center = centers[neighbor_idx]
            n_radius = radii[neighbor_idx]
            e_center = centers[expand_idx]
            e_radius = radii[expand_idx]
            
            # Distance
            dist = np.linalg.norm(e_center - n_center)
            max_possible = dist - (e_radius + n_radius)
            # We don't want to expand so that the new radius makes their circles tangent (allowing 0.01% margin)
            if max_possible < 1e-5:
                max_possible = 0  # can't expand
            max_expansion_amounts.append(max_possible)
        
        max_safe_growth = np.min(max_expansion_amounts)
        if max_safe_growth < 0:  # no safe expansion
            max_safe_growth = 0
        if max_safe_growth > 1e-3:  # cap to prevent over-expansion of very small radii
            max_safe_growth = 1e-3
        
        # Determine expansion amount
        expand_amount = np.clip(max_safe_growth * 0.8, 0, safety_margin)  # safety multiplier of 0.8

        # Expand this specific circle, and perturb nearby circles to allow system to adjust
        # We perturb nearby circles as well with smaller amounts to allow system to settle
        new_radii = radii.copy()
        new_radii[expand_idx] += expand_amount
        # Perturb neighbors with small expansion to allow adjustment (but only if growth remains safe)
        neighbors_to_perturb = np.argsort(dists[expand_idx, :])[:5]  # top 5 neighbors
        for neighbor in neighbors_to_perturb:
            if neighbor != expand_idx:
                dist_to_expand = dists[expand_idx, neighbor]
                neighbor_radius = radii[neighbor]
                expansion = min(
                    # Safe expansion amount considering existing radius
                    (dist_to_expand - (radii[expand_idx] + neighbor_radius)) * 0.3,  # 30% of available spacing
                    # Also consider safety constraint
                    safety_margin / (n - 1) * 0.5)  # 50% of per-circle safety margin

                # Only expand if the expansion remains safe
                if expansion > 0:
                    new_radii[neighbor] += expansion  # small growth to allow system to settle

        # Apply the new radii
        v_new = v.copy()
        v_new[2::3] = new_radii
        
        # Check if this configuration is safe before re-evaluation
        # This is a heuristic-based check to avoid invalidates
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = v_new[3*i] - v_new[3*j]
                dy = v_new[3*i+1] - v_new[3*j+1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < (v_new[3*i+2] + v_new[3*j+2]) - 1e-12:  # if overlap within tolerance
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            res_expanded = minimize(neg_sum_radii, v_new, method="SLSQP", 
                                   bounds=bounds, 
                                   constraints=cons, 
                                   options={
                                       "maxiter": 200, 
                                       "ftol": 1e-10, 
                                       "gtol": 1e-9, 
                                       "eps": 1e-6,
                                       "disp": False
                                   })
        else:
            # If the expansion is invalid, keep the old configuration
            res_expanded = res_reconfig

    # Final result
    v = res_expanded.x if res_expanded.success else res_reconfig.x if res_reconfig.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    sum_radii = float(radii.sum())
    return centers, radii, sum_radii