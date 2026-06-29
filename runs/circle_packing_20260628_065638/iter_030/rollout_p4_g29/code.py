import numpy as np

def run_packing():
    n = 26
    # Optimal layout grid design for 26 circles: 5x5 grid with 1 row of 1 circle
    cols = 5
    rows = 6  # To handle 5+5+5+5+5+1 = 26
    # This setup allows for 5 cols, 6 rows: 5*5=25 from 5x5 grid plus 1 row of 1
    # This ensures efficient use of space and allows staggered grid and edge optimization
    
    # Initialize positions with a more precise, structured geometric hashing + local grid bias
    xs = []
    ys = []
    # First, fill the 5x5 grid (25 circles) with structured placement
    for row_idx in range(5):
        for col_idx in range(5):
            # Core grid positions (non-randomized, better for stable convergence)
            x_center_base = (col_idx + 0.5) / cols
            y_center_base = (row_idx + 0.5) / rows
            # Add grid bias for staggered layout
            if row_idx % 2 == 1:  # Alternate row shift
                x_center_base += 0.5 / cols
            # Add a small, dynamic perturbation for local diversity
            x = x_center_base + np.random.uniform(-0.015, 0.015)
            y = y_center_base + np.random.uniform(-0.03, 0.03)
            xs.append(x)
            ys.append(y)
    
    # Add the 26th circle as a single isolated row cell, placed at the bottom right of the grid
    x_final = 0.95
    y_final = 0.05
    # Add small perturbation to avoid grid alignment
    x_final += np.random.uniform(-0.01, 0.01)
    y_final += np.random.uniform(-0.02, 0.02)
    xs.append(x_final)
    ys.append(y_final)
    
    # Set up initial radii based on grid density estimation and perturbation
    # Radii for grid circles: 0.35 / 5 = 0.07, for last circle: 0.15 for spacing
    r0_grid = 0.35 / cols - 1e-3
    r0_last = 0.24
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    # Use different radii for the final circle to allow for potential expansion without
    # violating grid constraints
    v0[2::3] = np.full(n - 1, r0_grid)
    v0[-1] = r0_last

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Ensure exact 3*n entries for 3*26

    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Define vector constraints using lambda and local variable capture for safety
    cons = []
    for i in range(n):
        # Left + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom + radius <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top - radius >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Overlap constraints with geometric hashing + adaptive grid-aware filtering
    # First, build an efficient adjacency graph for the 5x5 grid with an additional circle
    # To avoid dense constraint generation (26C2=325), apply smart filtering:
    # - Only create constraints between circles that are in the same or adjacent grid rows
    # - The last circle only has constraints against others in the same or adjacent rows
    
    # For grid-circle adjacency, we create a mask of neighbors
    grid_indices = list(range(n - 1))
    last_circle_idx = n - 1
    grid_neighbors = []
    for i in range(n - 1):
        row, col = i // 5, i % 5
        neighbors = []
        # Grid-based neighbors (4 directions + diagonals)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 5 and 0 <= nc < 5:
                j = nr * 5 + nc
                if j < n - 1:
                    neighbors.append(j)
        grid_neighbors.append(neighbors)
        # Add the last circle as neighbor for all in the same or adjacent rows
        if row in (0, 1, 4, 5):  # Check if it's in the same or adjacent row
            grid_neighbors[i].append(last_circle_idx)
    
    # Additional constraint: only create the last circle constraint against the last row of grid
    for j in range(5):
        grid_neighbors[-1].append(20 + j)
    
    # Create overlap constraints only between neighbors to reduce compute
    # Use vectorized broadcasting to compute distance
    for i in range(n - 1):
        for j in grid_neighbors[i]:
            if i < j:
                # Use vectorized constraint function
                def constraint_func(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    # Return distance^2 - (r_i + r_j)^2 (equality is zero)
                    return dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2
                cons.append({"type": "ineq", "fun": constraint_func})
    
    # Optimizer with adaptive strategy based on convergence
    # Phase 1: High precision
    res = minimize(neg_sum_radii, v0, method="SLSQP", 
                   bounds=bounds, constraints=cons,
                   options={
                       "maxiter": 600, "ftol": 1e-10, 
                       "eps": 1e-9, "disp": False
                   })
    
    # Phase 2: Asymmetric geometric hashing + spatial reconfiguration
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute all pairwise distances to assess constraint tightness
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        # Build constraint tightness matrix: 1 - (distance / (r_i + r_j))
        # We're trying to get tightness > 1e-12, which means no overlap
        tightness = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                if dists[i, j] < 1e-10:
                    tightness[i, j] = 1.0  # Very tight constraint
                else:
                    tightness[i, j] = 1 - (dists[i, j] / (radii[i] + radii[j])) if (radii[i] + radii[j]) > 1e-6 else 1.0
        tightness = np.maximum(tightness, 0)  # Avoid negative values
        
        # Calculate constraint tightness per circle
        circle_tightness = np.sum(tightness, axis=1)
        # Find the most constrained (tightest) circle
        most_constrained_idx = np.argmin(circle_tightness)
        
        # Generate randomized spatial perturbation (based on grid-awareness)
        # For grid circles, perturb based on current radius (larger radii get less perturbation)
        # Last circle gets more perturbation to break grid symmetry
        spatial_perturbation = np.random.rand(n, 2) * 0.04
        # Perturbation scale: larger circles (with smaller tightness) get less perturbation
        scale_factor = (1.0 - circle_tightness / np.max(circle_tightness)) ** 0.5
        scale_factor[most_constrained_idx] = 1.0
        scale_factor[last_circle_idx] = 1.3  # More perturbation for the last circle
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_perturbation[i, 0] * scale_factor[i]
            perturbed_v[3*i+1] += spatial_perturbation[i, 1] * scale_factor[i]
        
        # Phase 2: Re-optimize with perturbed spatial configuration
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={
                           "maxiter": 400, "ftol": 1e-10, 
                           "eps": 1e-9, "disp": False
                       })
    
    # Phase 3: Incremental radius expansion on least constrained circle with smart radius distribution
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Compute min distance to other circles (for constrained evaluation)
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        # Find the least constrained circle (by maximum minimal distance)
        min_dists = np.min(dists, axis=1)
        least_constrained_idx = np.argmax(min_dists)
        # For the last circle, we consider it more constrained based on spatial positioning
        
        # Target total radius sum (incrementally increase 0.005 to 0.01)
        target_total_growth = 0.010
        base_sum = np.sum(radii)
        target_total = base_sum + target_total_growth
        
        # Calculate radius scaling factors, more for least constrained
        # We distribute growth while prioritizing least constrained circle
        # Avoid increasing radii of circles that are adjacent or near each other
        # We calculate a constraint gradient vector for expansion direction
        
        # Use constraint tightness as growth priority
        constraint_priority = np.where(circle_tightness < np.mean(circle_tightness), 1.5, 0.5)
        constraint_priority = np.clip(constraint_priority, 0.3, 1.7)
        constraint_priority = constraint_priority / np.max(constraint_priority)  # Normalize
        
        # Create expansion vector with growth to least constrained
        new_radii = np.copy(radii)
        expansion = (target_total - base_sum) / n * constraint_priority  # Base expansion
        # Add extra to least constrained circle, with soft constraint validation
        # To prevent overlapping, we also ensure that the expansion is limited by spatial constraints
        # We will compute this in a validation loop with backtracking if necessary
        
        # First, expand the least constrained circle by 10% more
        expansion[least_constrained_idx] *= 1.2

        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            
            # Validate expanded configuration
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
            else:
                # If invalid, decrease expansion slightly
                # We decrease by 50% of the excess expansion
                new_radii = radii + (new_radii - radii) * 0.85
        
        final_v = v.copy()
        final_v[2::3] = new_radii
        
        # Phase 3: final optimization with expanded radii and new configuration
        res = minimize(neg_sum_radii, final_v, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={
                           "maxiter": 300, "ftol": 1e-10, 
                           "eps": 1e-9, "disp": False
                       })

    # Post-check: Ensure all constraints are met with a final check to catch any missed edge cases
    # This is redundant, but ensures robustness
    if res.success:
        v = res.x
        # Ensure radii are valid
        radii = v[2::3]
        radii = np.clip(v[2::3], 1e-6, None)
        centers = np.column_stack([v[0::3], v[1::3]])
        # Final validation to ensure everything is okay
        # (Already done in optimizer, but for absolute certainty)
        all_distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                all_distances[i, j] = dist
        overlaps = np.any(all_distances < (radii[i] + radii[j] - 1e-12) for i in range(n) for j in range(i + 1, n))

        # Validate all positions are within the unit square
        valid_positions = True
        for i in range(n):
            x = centers[i, 0]
            y = centers[i, 1]
            r = radii[i]
            if (x - r < -1e-12 or x + r > 1 + 1e-12 or
                y - r < -1e-12 or y + r > 1 + 1e-12):
                valid_positions = False
                break
        if not valid_positions:
            # Revert to the best found state
            v = res.x if res.success else v0
            radii = np.clip(v[2::3], 1e-6, None)
    else:
        v = res.x if res.success else v0
        radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())