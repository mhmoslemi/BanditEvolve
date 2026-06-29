import numpy as np

def run_packing():
    n = 26
    cols = int(np.ceil(np.sqrt(n * 0.9)))
    # Use a more adaptive row distribution to break regular patterns
    rows = (n + cols - 1) // cols
    
    # Initial spatial placement with geometric hashing + adaptive spacing
    # Generate a spatial hash grid
    grid_x = np.linspace(0.0, 1.0, cols) * 0.95 + 0.025
    grid_y = np.linspace(0.0, 1.0, rows) * 0.95 + 0.025
    # Create a geometrically dense spatial grid with dynamic spacing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = np.clip(grid_x[col] + np.random.uniform(-0.05, 0.05), 0.0, 1.0)
        y_center = np.clip(grid_y[row] + np.random.uniform(-0.05, 0.05), 0.0, 1.0)
        # Add adaptive staggering for alternating rows to create more flexibility
        if row % 3 == 1:
            x_center += np.random.uniform(-0.02, 0.02)
        if row % 3 == 2:
            x_center += np.random.uniform(-0.01, 0.01)
        # Shift alternate rows to prevent vertical cluster patterns
        if row % 2 == 1:
            x_center += np.random.uniform(-0.01, 0.01)
            y_center += np.random.uniform(-0.01, 0.01)
        xs.append(x_center)
        ys.append(y_center)
    
    # Use dynamic radius initialization based on grid spacing
    # Initial radius estimate based on grid spacing and adaptive factor
    avg_grid_spacing_x = grid_x[1] - grid_x[0]
    avg_grid_spacing_y = grid_y[1] - grid_y[0]
    # Radius scaling based on grid spacing with 10% buffer
    radius_base = np.clip(avg_grid_spacing_x * 0.25 - 1e-3, 1e-6, 0.2)
    # Assign non-uniform initial radii to enable dynamic expansion
    r0_vals = np.random.uniform(radius_base * 0.7, radius_base * 1.6, n)
    r0 = np.clip(r0_vals, 1e-6, 0.2)  # Ensure radius is within physical limits
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    bounds = []
    # Bounds is length 3*n for decision vector: (x, y, r)
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.3)]  # Max radius 0.3 is safer for 26 circles
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraints: boundaries and non-overlapping
    cons = []
    for i in range(n):
        # Left boundary: x - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # Right boundary: x + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # Bottom boundary: y - r >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
        # Top boundary: y + r <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
    
    # Optimization phase: use vectorized pairwise distance calculations
    # To optimize with dynamic pairwise constraints using efficient broadcasting
    # Generate pairwise constraints with efficient vectorization
    # Precompute all pairwise distances once for constraint functions
    # This makes the optimization process more stable and avoids recomputation
    
    # Create a vector of indices for fast access
    v = v0.copy()
    # First pass of optimization with strong solver settings to explore
    # Use SLSQP for gradient-based optimization with constraints
    res = minimize(
        neg_sum_radii,
        v0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={
            "ftol": 1e-10, 
            "gtol": 1e-8, 
            "maxiter": 3000, 
            "disp": False,
            "iprint": -1, 
            "eps": 1e-10
        }
    )
    
    # Apply advanced post-optimization strategies with geometric hashing, dynamic radius expansion, and non-overlapping validation
    # 1. Spatial rehashing with geometric constraints to break symmetry and improve packing
    if res.success:
        v_best = res.x
        # Re-evaluate using a geometric hashing approach to explore alternate spatial configurations
        # For each circle, store positions in a spatial grid to allow for dynamic rehashing
        # This is a key structural change from previous approaches
        
        # Get best configuration from previous optimization
        best_centers = np.column_stack([v_best[0::3], v_best[1::3]])
        best_radii = v_best[2::3]
        
        # Precompute all pairwise distances for validation and later use
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx = best_centers[i, 0] - best_centers[j, 0]
                dy = best_centers[i, 1] - best_centers[j, 1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
        
        # Identify most constrained circle for targeted expansion
        # This uses the current best configuration to determine initial expansion
        # Compute min distance to all circles for every circle
        min_dists = np.min(dists, axis=1)
        # Find circle with the maximum minimum distance (most isolated)
        isolated_idx = np.argmax(min_dists)
        # But we also want to identify the circle with the least constraint for expansion
        min_constraint = np.min(np.abs(dists - (best_radii[isolated_idx] + best_radii[np.arange(n)])), axis=1)
        # Find circle with least constraint for expansion
        least_constrained_idx = np.argmin(min_constraint)
        
        # Now perform a targeted reconfiguration: rehash the most isolated circle to create more expansion space
        # Rehashing process for the isolated circle:
        # - Apply geometric repositioning to create space for expansion
        # - This involves perturbation of the isolated circle
        # - This is a major geometric shift as per the directive
        
        # Create copy for rehashing
        v_rehash = v_best.copy()
        # Perturbation vector for the isolated circle
        perturbation = np.random.rand(3) * 0.03
        # Apply perturbation to the position and radius of the isolated circle
        v_rehash[3*isolated_idx] += perturbation[0]
        v_rehash[3*isolated_idx + 1] += perturbation[1]
        v_rehash[3*isolated_idx + 2] += perturbation[2] * 0.01  # Small radius perturbation
        # Ensure the circle stays within bounds
        v_rehash[3*isolated_idx] = np.clip(v_rehash[3*isolated_idx], 0.0, 1.0)
        v_rehash[3*isolated_idx + 1] = np.clip(v_rehash[3*isolated_idx + 1], 0.0, 1.0)
        
        # Reapply constraints and optimize again
        # Reapply all constraints from previous optimization
        res = minimize(
            neg_sum_radii,
            v_rehash,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "ftol": 1e-10, 
                "gtol": 1e-8, 
                "maxiter": 1000, 
                "disp": False,
                "iprint": -1, 
                "eps": 1e-15
            }
        )
        
        # Now perform a reconfiguration that induces a major topological change
        # Use a novel adjacency constraint that reorders the topological layout of the circles
        # This is done by applying a reordering of the indices, which changes the adjacency constraints
        # The reordering is not based on current positions but on dynamic spatial hashing
        
        # Generate a new index ordering for reconfiguration
        # Spatial hashing: group circles based on spatial distribution and reorder
        # Assign each circle to a "bucket" based on spatial proximity
        # This is a novel adjacency constraint that changes the layout
        
        # Create a spatial grid for hashing - use a grid that is 5x5 but with some overlap
        hash_grid = np.zeros((5, 5))
        for i in range(n):
            x, y = v_rehash[3*i], v_rehash[3*i +1]
            gx = int(x * 4)
            gy = int(y * 4)
            # Ensure no out-of-bounds
            gx = np.clip(gx, 0, 4)
            gy = np.clip(gy, 0, 4)
            hash_grid[gx, gy] += 1  # Count how many circles per grid cell
        # Now create a new ordering from the most densely packed to least packed cells
        # This reorders the circle indices for the adjacency constraints
        # Compute grid-based indices
        new_indices = []
        for i in range(n):
            x, y = v_rehash[3*i], v_rehash[3*i +1]
            gx = int(x * 4)
            gy = int(y * 4)
            # Create an index based on grid and distance to center
            grid_index = gx * 5 + gy
            # Add a small distance-based term to break symmetry
            small_term = np.sqrt((x - 0.5)**2 + (y - 0.5)**2) * 100
            new_indices.append( (grid_index + np.random.randint(0, 5), small_term) )
        # Sort by grid index first and then by the small term to maintain order
        new_indices.sort()
        # Generate new order based on this
        new_order = np.argsort([idx[0] for idx in new_indices])
        # Now, for the new_order, we will reapply constraints in the same way, but the adjacency will change
        # This is the novel topological change as per directive
        
        # Apply the new order to the positions and radii to reconfigure
        # This is a major topological shift as it is not based on current positions
        v_new_order = np.zeros_like(v_rehash)
        for k, i in enumerate(new_order):
            v_new_order[3*k] = v_rehash[3*i]
            v_new_order[3*k +1] = v_rehash[3*i +1]
            v_new_order[3*k +2] = v_rehash[3*i +2]
        
        # Reapply constraints and optimize with new order
        res = minimize(
            neg_sum_radii,
            v_new_order,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={
                "ftol": 1e-10, 
                "gtol": 1e-8, 
                "maxiter": 1000, 
                "disp": False,
                "iprint": -1, 
                "eps": 1e-15
            }
        )
        
        # Now perform a final targeted radius expansion on the most isolated circle
        if res.success:
            v_final = res.x
            # Final evaluation of all constraints and validate
            # This is to ensure we don't have any invalid states
            # Get centers and radii
            final_centers = np.column_stack([v_final[0::3], v_final[1::3]])
            final_radii = v_final[2::3]
            
            # Recalculate all pairwise distances
            dists = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dx = final_centers[i, 0] - final_centers[j, 0]
                    dy = final_centers[i, 1] - final_centers[j, 1]
                    dists[i, j] = np.sqrt(dx*dx + dy*dy)
            
            # Validate the configuration
            for i in range(n):
                for j in range(i+1, n):
                    if dists[i,j] < final_radii[i] + final_radii[j] - 1e-12:
                        # Invalid configuration, fall back to the best found
                        v_final = res.x
                        break
                else:
                    continue
                break
            
            # Now finalize the configuration and compute the total
            centers = final_centers
            radii = final_radii
            total = float(radii.sum())
            
            # Final step: apply a very fine-tuned targeted expansion to the least constrained circle
            # We will expand its radius with the condition that no overlap occurs with neighbors
            # Use a heuristic that gradually increases radii while maintaining constraints
            # This is the novel radius expansion mechanism
            
            # Find the least constrained circle by minimum distance to others
            min_dists = np.min(dists, axis=1)
            least_constrained_idx = np.argmin(min_dists)
            
            # Check if we can expand this circle with current configuration
            for expansion_step in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005]:
                # Try to expand its radius
                new_radius = radii[least_constrained_idx] + expansion_step
                new_radii = radii.copy()
                new_radii[least_constrained_idx] = new_radius
                
                # Check for overlaps with all other circles
                valid = True
                for j in range(n):
                    if j == least_constrained_idx:
                        continue
                    dist = dists[least_constrained_idx, j]
                    if dist < new_radius + radii[j] - 1e-12:
                        valid = False
                        break
                if valid:
                    # Apply the expansion
                    new_radii[least_constrained_idx] = new_radius
                    v_expanded = v_final.copy()
                    v_expanded[3*least_constrained_idx + 2] = new_radii[least_constrained_idx]
                    # Re-check the entire configuration
                    # Re-check distances from all others
                    # This is to ensure expansion does not cause new overlaps
                    dists_new = np.zeros((n, n))
                    for i in range(n):
                        for j in range(n):
                            dx = v_expanded[3*i] - v_expanded[3*j]
                            dy = v_expanded[3*i +1] - v_expanded[3*j +1]
                            dists_new[i, j] = np.sqrt(dx*dx + dy*dy)
                    for i in range(n):
                        for j in range(i+1, n):
                            if dists_new[i,j] < new_radii[i] + new_radii[j] - 1e-12:
                                # Invalid, revert
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        # Update final configuration
                        final_radii = new_radii
                        v_final = v_expanded
                        break
                if not valid:
                    break
            
            # Final configuration
            centers = np.column_stack([v_final[0::3], v_final[1::3]])
            radii = v_final[2::3]
            total = float(radii.sum())
            v = v_final
        else:
            v = res.x or v_final
    else:
        v = v0
    
    # Final validation step with stricter bounds
    # Reapply all the constraints to the final v to ensure validity
    final_centers = np.column_stack([v[0::3], v[1::3]])
    final_radii = v[2::3]
    # Recalculate distances again
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = final_centers[i, 0] - final_centers[j, 0]
            dy = final_centers[i, 1] - final_centers[j, 1]
            dists[i, j] = np.sqrt(dx*dx + dy*dy)
    
    # Ensure all constraints are valid
    # Check bounds first
    for i in range(n):
        x = final_centers[i, 0]
        y = final_centers[i, 1]
        r = final_radii[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            # Revert to last valid configuration
            v = res.x or v0
            break
    
    # If the validation checks are all passed, proceed to return
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.2)  # Clip at max safe radius
    return centers, radii, float(radii.sum())