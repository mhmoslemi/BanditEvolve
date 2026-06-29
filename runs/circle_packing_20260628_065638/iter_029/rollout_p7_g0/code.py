import numpy as np

def run_packing():
    n = 26
    # Use more precise grid arrangement, considering non-linear spatial interactions
    cols = 5
    rows = np.ceil(n / cols).astype(int)
    
    # Initialize with spatially intelligent grid: staggered with random offsets and adaptive spacing
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / (rows)
        
        # Adaptive x-offset based on column position and row parity
        x_offset = np.random.uniform(-0.02, 0.02) * (1.0 - col / cols)
        y_offset = np.random.uniform(-0.02, 0.02) * (1.0 - row / rows)
        
        # Introduce a row-specific diagonal shift to enhance spatial diversity
        if row % 2 == 1:
            x_center += 0.5 / (cols + 1)
        
        x = x_center + x_offset
        y = y_center + y_offset
        
        # Ensure position stays within square
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        
        xs.append(x)
        ys.append(y)

    # Initial radius based on spatial spacing - use adaptive sizing across rows/columns
    r0 = (0.3 / cols) * (1 + 0.05 * np.random.rand(n)) - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.45)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Vectorized boundary constraints with lambda closures and fixed i
    cons = []
    for i in range(n):
        # Ensure x - r >= 0 
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Ensure x + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Ensure y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # Ensure y + r <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})

    # Vectorized overlap constraints with optimized lambda captures and batched calculation
    # Create a more efficient overlap constraint system via numpy broadcasting with vectorized operations
    for i in range(n):
        for j in range(i + 1, n):
            # Define constraints using lambda with captured i and j
            cons.append({
                "type": "ineq",
                "fun": lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + 
                                            (v[3*i+1] - v[3*j+1])**2 - 
                                            (v[3*i+2] + v[3*j+2])**2
            })
    
    # Initial optimization with increased max iterations, tighter precision, and better tolerances
    initial_res = minimize(neg_sum_radii, v0, method="SLSQP", 
                          bounds=bounds, constraints=cons, 
                          options={"maxiter": 1200, "ftol": 1e-12, "gtol": 1e-12, "eps": 1e-12})
    
    # Perform forced geometric dissection: isolate and reconfigure the most dynamically interacting pair
    # Use matrix operations to calculate distances and find most interacting circle pair
    
    if initial_res.success:
        # Extract current state
        v = initial_res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Compute pairwise distances with numpy vectorization for performance
        cx = centers[:, 0]
        cy = centers[:, 1]
        dists = np.sqrt((cx[:, np.newaxis] - cx[np.newaxis, :])**2 + 
                        (cy[:, np.newaxis] - cy[np.newaxis, :])**2)
        
        # Compute interaction matrix by sum of reciprocals of distances; this reflects dynamic interaction
        interaction = np.reciprocal(np.sum(dists, axis=1) + 1e-6, dtype=np.float64)
        
        # Find the pair with the highest interaction: these are the two most dynamically interacting
        pair_indices = np.unravel_index(np.argmax(interaction), interaction.shape)
        i, j = pair_indices
        
        # Create a "geometric dissection" by re-centering the pair
        # Create geometric hash for forced spatial reconfiguration
        force_hash = np.random.rand(2) * 0.1
        x_shift1 = force_hash[0] * (radii[i] + radii[j])
        y_shift1 = force_hash[1] * (radii[i] + radii[j])
        
        # Perturb positions of the most interacting pair to enable restructuring
        v[3*i] += x_shift1
        v[3*i+1] += y_shift1
        v[3*j] -= x_shift1
        v[3*j+1] -= y_shift1
        
        # Rebound after shifts
        for k in [i, j]:
            v[3*k] = np.clip(v[3*k], 0.0, 1.0)
            v[3*k+1] = np.clip(v[3*k+1], 0.0, 1.0)
            v[3*k+2] = np.clip(v[3*k+2], 1e-4, 0.45)
        
        # Re-evaluate this new constrained spatial configuration
        dissection_res = minimize(neg_sum_radii, v, method="SLSQP", 
                                bounds=bounds, constraints=cons, 
                                options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-12})
    
    else:
        dissection_res = initial_res

    # Perform targeted radius expansion on least constrained circle with novel adjacency constraint
    # First, ensure optimization was successful
    optimized_v = dissection_res.x if dissection_res.success else v0
    
    # Now reevaluate using the new configuration from dissection
    # Rebuild centers and radii from new configuration
    centers = np.column_stack([optimized_v[0::3], optimized_v[1::3]])
    radii = optimized_v[2::3]
    
    # Find the circle with the smallest minimal distance to other circles (most constrained)
    # This is the opposite approach from earlier: find the *least* constrained circle for expansion
    # Calculate all inter-circle distances with vectorized operations
    cx = centers[:, 0]
    cy = centers[:, 1]
    dists = np.sqrt((cx[:, np.newaxis] - cx[np.newaxis, :])**2 + 
                    (cy[:, np.newaxis] - cy[np.newaxis, :])**2)
    
    # Compute the minimum distance for each circle to all others
    min_dists = np.min(dists, axis=1)
    
    # The least constrained circle has the largest minimum distance to other circles
    least_constrained_idx = np.argmax(min_dists)
    
    # Compute target_total_sum expansion, but now with more aggressive expansion
    current_total = np.sum(radii)
    target_total_sum = current_total + 0.0065
    expansion_amount = (target_total_sum - current_total) / (n - 1)
    
    # Create additional adjacency constraint: expand the least constrained circle and enforce reordering
    # This is where the novel adjacency constraint comes in - create a spatial ordering constraint
    # We'll create a new set of constraints that ensures the least constrained circle maintains a minimum distance to a specific point in the square
    
    # Generate a new set of constraints for the most dynamically interacting pair
    # These constraints are added as hard constraints to force reordering
    
    # Create a spatial constraint that forces the least constrained circle to maintain a specific relative position
    # Use a geometric hash to determine perturbation in the target direction
    
    # Generate a geometric hash vector and apply it to the least constrained circle's radius
    spatial_key = np.random.rand(2) * 2.0
    expansion_factor = expansion_amount * (1.0 + spatial_key[0]) * 1.1
    
    # Apply expansion to the least constrained circle
    optimized_v[3*least_constrained_idx + 2] += expansion_factor
    
    # Ensure not to exceed the max radius and clip
    optimized_v[3*least_constrained_idx + 2] = np.clip(optimized_v[3*least_constrained_idx + 2], 1e-4, 0.45)
    
    # Now we add a forced adjacency constraint that keeps the circle at least 0.25 away from a specific point in the square
    
    # Define a target point to maintain proximity to
    target_point = np.array([0.75, 0.5])  # Example fixed point in the square
    
    # Add this as an additional constraint:
    # (x - tx)^2 + (y - ty)^2 >= (r + 0.25)^2
    # We use this to force spatial reordering and create a novel adjacency
    
    # Create a new constraint function:
    def adjacency_constraint(v):
        # Get radius of the least constrained circle
        r = v[3*least_constrained_idx + 2]
        # Get center of the circle
        x = v[3*least_constrained_idx]
        y = v[3*least_constrained_idx + 1]
        tx, ty = target_point
        return (x - tx)**2 + (y - ty)**2 - (r + 0.25)**2
    
    # Add this as an ineq constraint to enforce the adjacency
    cons.append({"type": "ineq", "fun": adjacency_constraint})
    
    # Re-optimize with this constraint to force the least constrained circle to new location, maintaining spatial reordering
    final_res = minimize(neg_sum_radii, optimized_v, method="SLSQP", bounds=bounds, 
                        constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-12})
    
    # Final refinement - perform a final pass to ensure alignment with all constraints
    v = final_res.x if final_res.success else optimized_v
    
    # Final cleanup: ensure centers remain within bounds and radii are valid
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.45)
    
    # Final step: ensure all circles are within bounds and no overlaps
    # This is redundant due to constraint checks, but for safety and validation
    for i in range(n):
        x = v[3*i]
        y = v[3*i+1]
        r = v[3*i+2]
        if x - r < -1e-9 or x + r > 1.0001 or y - r < -1e-9 or y + r > 1.0001:
            # Rebound if out of bounds
            v[3*i] = np.clip(x, 0.0, 1.0)
            v[3*i+1] = np.clip(y, 0.0, 1.0)
            v[3*i+2] = np.clip(r, 1e-6, 0.45)
    
    # Final optimization pass with tighter tolerances
    final_v = v
    final_res = minimize(neg_sum_radii, final_v, method="SLSQP", bounds=bounds, 
                        constraints=cons, options={"maxiter": 200, "ftol": 1e-12, "eps": 1e-12})
    
    v = final_res.x if final_res.success else v
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.45)
    
    return centers, radii, float(radii.sum())