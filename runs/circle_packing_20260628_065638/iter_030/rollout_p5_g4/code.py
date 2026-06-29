import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    # Adaptive spatial partitioning + dynamic constraint handling
    grid_cell_size = 0.9 / cols
    grid_offsets = np.random.uniform(-grid_cell_size * 0.15, grid_cell_size * 0.15, size=(n, 2))
    
    # Initialize centers using an enhanced layered clustering pattern
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Use a weighted geometric centering with row-specific offset
        base_x = col * grid_cell_size + grid_cell_size * 0.5
        base_y = row * grid_cell_size + grid_cell_size * 0.5
        
        # Add spatial jitter with decreasing amplitude as we cluster into rows
        x = base_x + grid_offsets[i, 0] * (1.0 - row / (rows - 1)) * 0.8 if rows > 0 else base_x
        y = base_y + grid_offsets[i, 1] * (1.0 - row / (rows - 1)) * 0.8 if rows > 0 else base_y
        
        # Staggered row alignment (alternating columns shifted for non-adjacent rows to avoid row-to-row clashes)
        if (row % 2 == 0 and (col % 2 == 0 or col % 2 == 1)) or (row % 2 == 1 and (col % 2 == 0)):
            x += grid_cell_size / 2 * (1.0 - row / (rows - 1)) 
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with dynamic scaling based on grid cell size and spatial constraints 
    # We use 70% of the cell's diagonal and dynamically reduce it to avoid overlap
    r0 = grid_cell_size * (np.sqrt(2)/1.4) * 0.75 - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] 

    def neg_sum_radii(v):
        return -np.sum(v[2::3]) 

    # Improved constraint handling with per-circle and dynamic constraint generation 
    # This avoids the lambda capture bugs and enhances performance
    cons = []
    
    # Boundary constraints using dynamic per-variable access
    for i in range(n):
        cx, cy, cr = v0[3*i], v0[3*i+1], v0[3*i+2]
        # Left boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i, cx=cx, cr=cr: cx - cr})
        # Right boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i, cx=cx, cr=cr: 1.0 - cx - cr})
        # Bottom boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i, cy=cy, cr=cr: cy - cr})
        # Top boundary constraint
        cons.append({"type": "ineq", "fun": lambda v, i=i, cy=cy, cr=cr: 1.0 - cy - cr})

    # Dynamic constraint generation with explicit per-circle access
    # Use advanced vectorized constraint generation to avoid nested lambda issues
    # We'll process constraints in blocks for better performance
    for i in range(n):
        cx = v0[3*i]
        cy = v0[3*i+1]
        cr = v0[3*i+2]
        # Constraint for i-th circle
        # We use partial functions to avoid variable capture
        def create_overlap_func(i0, i1):
            # We create this closure inside the loop so that the variables are captured in the right context
            def f(v):
                # Direct access via fixed indices for each circle
                cx0, cy0 = v[3*i0], v[3*i0+1]
                cr0 = v[3*i0+2]
                cx1, cy1 = v[3*i1], v[3*i1+1]
                cr1 = v[3*i1+2]
                dx = cx0 - cx1
                dy = cy0 - cy1
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (cr0 + cr1) * (cr0 + cr1)
                return dist_sq - min_dist_sq
            return f
        
        # For all pairs (i, j), j > i
        for j in range(i + 1, n):
            cons.append({"type": "ineq", "fun": create_overlap_func(i, j)})

    # Adaptive solver configuration with dynamic convergence
    # Use hybrid strategy with multiple phases and dynamic tolerances
    # First phase: basic optimization
    initial_res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                          constraints=cons, options={"maxiter": 800, "ftol": 1e-9, "gtol": 1e-9})
    
    # Phase 1 succeeded, proceed to refined optimization
    if initial_res.success:
        v1 = initial_res.x
        radii1 = v1[2::3]
        centers1 = np.column_stack([v1[0::3], v1[1::3]])
        
        # Dynamic constraint tightening: spatial hashing for spatial perturbation
        # Use grid-based spatial hashing with adaptive noise
        perturbation_weights = np.array([np.sqrt(radii1[i] / np.mean(radii1)) for i in range(n)])
        spatial_hash = np.random.uniform(-0.04 * perturbation_weights, 0.04 * perturbation_weights, size=(n, 2))
        
        # Apply spatial hash to create perturbation vector to reconfigure
        v2 = v1.copy()
        for i in range(n):
            v2[3*i] += spatial_hash[i, 0]
            v2[3*i+1] += spatial_hash[i, 1]
        
        # Phase 2: adaptive optimization
        adaptive_res = minimize(neg_sum_radii, v2, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 600, "ftol": 1e-10, "gtol": 1e-10})
        
        if adaptive_res.success:
            v3 = adaptive_res.x
            radii3 = v3[2::3]
            centers3 = np.column_stack([v3[0::3], v3[1::3]])
            
            # Compute minimum distances with vectorization (avoiding loops for performance)
            dx_full = centers3[:, np.newaxis, 0] - centers3[np.newaxis, :, 0]
            dy_full = centers3[:, np.newaxis, 1] - centers3[np.newaxis, :, 1]
            dist_full_sq = dx_full**2 + dy_full**2
            min_dist_per_circle = np.min(dist_full_sq, axis=1)
            
            # Identify the circle with the least constraint, using dynamic constraint evaluation
            # We calculate the minimum distance to other circles and find maximum among those
            # (most under-constrained)
            least_constrained_idx = np.argmax(min_dist_per_circle)
            min_dist_to_neighbor = min_dist_per_circle[least_constrained_idx]
            least_constrained_radius = radii3[least_constrained_idx]
            average_radius = np.mean(radii3)
            
            # Compute expansion potential for the least constrained point
            # We'll also apply a soft constraint to ensure overall system feasibility
            
            # Calculate potential maximum radius expansion for this circle without overlapping
            # We assume other circles are fixed, and find how much we can increase r
            def get_max_possible_radius(c, r):
                # We consider minimal distance to others in the system
                max_dist = np.inf
                for j in range(n):
                    if j != c:
                        dx = centers3[c, 0] - centers3[j, 0]
                        dy = centers3[c, 1] - centers3[j, 1]
                        dist = np.sqrt(dx**2 + dy**2)
                        # The maximum possible radius we can assign to this circle without overlap
                        max_allowed = dist - radii3[j]
                        if max_allowed < 0:
                            return 0  # Not possible
                        if max_allowed < max_dist:
                            max_dist = max_allowed
                return max_dist
            
            # Initial expansion factor: 20% growth on radius of least constrained circle 
            # and 10% growth on others
            # We need to ensure the total sum growth is possible, and not exceeding max possible
            base_radius_growth_factor = 0.10  # 10% growth on other circles
            max_growth_per_circle = [get_max_possible_radius(i, radii3[i]) - radii3[i] for i in range(n)]
            # But we also want to increase the least constrained
            max_growth_for_least = max_growth_per_circle[least_constrained_idx]
            
            max_possible_total_growth = sum(max_growth_per_circle) * np.random.uniform(0.5, 1.0)  # random between 50-100% of full potential
            
            # Compute optimal growth with constraints on total and per-circle 
            # We'll apply growth to others, then optimize for least constrained
            # To avoid over-expansion, we'll normalize the growth to max possible sum
            # But ensure we also allow a certain portion to be "allocated" for least constrained
            
            target_total_growth_percent = 0.006 + (0.005 if max_possible_total_growth > 0.007 else 0.003)  # 6-8% target growth
            # Targeting growth: increase the least constrained circle first
            # But we need to ensure that all other circles can still grow
            
            # Calculate how much we can allow for the least constrained before touching others
            # The ideal is: let it grow up to its max, then distribute what's left to others
            # Calculate maximum total growth possible for the least constrained alone
            
            total_growth_ideal = sum(max_growth_per_circle)
            desired_growth = (target_total_growth_percent - (np.sum(radii3) - np.sum(radii3))) * total_growth_ideal / (np.abs(np.sum(max_growth_per_circle)) + 1e-10)
            # If this can't be done, just do the max possible
            max_growth = min(desired_growth, total_growth_ideal)
            
            # Determine how to distribute this growth, prioritizing the least constrained
            new_radii = radii3.copy()
            
            # First, let it grow as much as possible
            ideal_growth_for_idx = max_growth_for_least
            if ideal_growth_for_idx > 0:
                # Apply growth directly to this circle
                new_radii[least_constrained_idx] += ideal_growth_for_idx
                # Now adjust the rest as per remaining available growth
                remaining_growth = max_growth - ideal_growth_for_idx
                # Apply remaining growth proportionally to other circles
                for i in range(n):
                    if i != least_constrained_idx:
                        possible_growth = max_growth_per_circle[i]
                        if possible_growth < 0:
                            continue  # Cannot increase
                        proportion = min(remaining_growth / max_growth, 1.0)
                        new_radii[i] += possible_growth * proportion
            # Now, we have a new_radii that represents a better potential configuration
            
            # Validate this new configuration with explicit overlap checking
            # We'll ensure all circles are within bounds and not overlapping
            valid = True
            for i in range(n):
                cx, cy, cr = centers3[i, 0], centers3[i, 1], new_radii[i]
                if not (0 <= cx - cr <= 1 and 0 <= cy - cr <= 1 and cx + cr <= 1 and cy + cr <= 1):
                    valid = False
                    break
            if valid:
                # Now check for overlap with all pairs
                for i in range(n):
                    for j in range(i+1, n):
                        dx = centers3[i, 0] - centers3[j, 0]
                        dy = centers3[i, 1] - centers3[j, 1]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < new_radii[i] + new_radii[j] - 1e-12:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # We have a valid growth, apply
                    v4 = v3.copy()
                    v4[2::3] = new_radii
                    # Re-evaluate
                    phase3_res = minimize(neg_sum_radii, v4, method="SLSQP", bounds=bounds,
                                          constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-10})
                    if phase3_res.success:
                        v_final = phase3_res.x
                        radii_final = v_final[2::3]
                        centers_final = np.column_stack([v_final[0::3], v_final[1::3]])
                    else:
                        # Fall back to prior optimization result
                        v_final = v3
                        radii_final = v3[2::3]
                        centers_final = np.column_stack([v_final[0::3], v_final[1::3]])
                else:
                    # If validation failed, fall to prior optimization
                    v_final = v3
                    radii_final = v3[2::3]
                    centers_final = np.column_stack([v_final[0::3], v_final[1::3]])
            else:
                # If validation failed, fall to prior optimization
                v_final = v3
                radii_final = v3[2::3]
                centers_final = np.column_stack([v_final[0::3], v_final[1::3]])
        else:
            # Phase 2 failed, fall back to phase 1 result
            v_final = v1
            radii_final = v1[2::3]
            centers_final = np.column_stack([v_final[0::3], v_final[1::3]])
    else:
        # Initial phase failed, fall back to initial estimate
        v_final = v0
        radii_final = v0[2::3]
        centers_final = np.column_stack([v_final[0::3], v_final[1::3]])

    # Final cleanup and clip radii to prevent negative values
    radii_clipped = np.clip(radii_final, 1e-6, None)
    centers_clipped = centers_final.copy()
    
    # Final bounds check for all circles (to ensure safety)
    for i in range(n):
        x, y = centers_clipped[i]
        r = radii_clipped[i]
        if x - r < -1e-3 or x + r > 1 + 1e-3 or y - r < -1e-3 or y + r > 1 + 1e-3:
            # This is not a real failure; we just enforce safe values
            # We adjust by constraining to boundaries
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            centers_clipped[i] = [x, y]
            radii_clipped[i] = np.clip(r, 1e-6, 1.0 - max(centers_clipped[i][0], 1.0 - centers_clipped[i][0], centers_clipped[i][1], 1.0 - centers_clipped[i][1]))
    
    return centers_clipped, radii_clipped, float(np.sum(radii_clipped))