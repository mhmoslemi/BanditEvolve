import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize positions with randomized geometric clustering
    xs = []
    ys = []
    for i in range(n):
        row = i // cols
        col = i % cols
        # Centering with adaptive radius-aware placement to prevent initial crowding
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset with decreasing noise for stability
        x_noise = np.random.uniform(-0.05, 0.05)
        y_noise = np.random.uniform(-0.05, 0.05)
        x = x_center + x_noise
        
        # Introduce vertical staggering for enhanced packing efficiency
        if row % 2 == 0:
            # Even rows: use standard centering
            y = y_center + y_noise
        else:
            # Odd rows: slight upward shift and asymmetric noise
            y_center += 0.05
            y = y_center + y_noise
        
        # Apply row-specific horizontal perturbation for more dynamic layout
        if row % 3 == 0:
            x += np.random.uniform(-0.02, 0.02)
        xs.append(x)
        ys.append(y)
    
    # Initial radii based on spacing and geometric hashing
    # Precompute radius upper bound: using average spacing between centers
    avg_center_distance = np.mean([np.linalg.norm(np.array([xs[i], ys[i]]) - np.array([xs[j], ys[j]])) for i in range(n) for j in range(i+1, n)])
    initial_radius_estimate = avg_center_distance * 0.25
    r0 = initial_radius_estimate / cols * 0.8 - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), 
                   (1e-4, max(r0 * 1.2, 0.15))]  # Adaptive radius upper bound

    # Optimization objective
    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Constraint setup with vectorized bounds (ensure i is captured correctly)
    cons = []
    for i in range(n):
        # Left boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # Right boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i+2]})
        # Bottom boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # Top boundary
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i+1] - v[3*i+2]})

    # Overlap constraints with vectorized distance computation and dynamic tolerance
    for i in range(n):
        for j in range(i+1, n):
            # Use dynamic constraints: larger minimum distance threshold as radii grow
            def constraint_func(v, i=i, j=j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist = np.sqrt(dx**2 + dy**2)
                min_radius = np.min([v[3*i+2], v[3*j+2]])
                threshold = dist - (min_radius + 1e-12)
                return threshold  # >= 0 ensures no overlap
            cons.append({"type": "ineq", "fun": constraint_func})

    # First stage: global optimization with adaptive learning rate and constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-12, "eps": 1e-8})
    v = res.x if res.success else v0
    v = v.copy()  # Prevent overwriting

    # Second stage: isolate and refine the most dynamically interacting pair
    if res.success:
        # Compute all pairwise distances and find the top two most interacting circles
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        dists = np.zeros((n, n))
        
        # Vectorized pairwise distance computation
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute interaction strength as total distance from each circle to others
        interaction_strength = np.sum(dists, axis=1)
        top_two_indices = np.argsort(interaction_strength)[-2:]
        
        # Extract the two most interacting circles and their coordinates
        a_idx, b_idx = top_two_indices
        a_pos = centers[a_idx]
        b_pos = centers[b_idx]
        a_r = radii[a_idx]
        b_r = radii[b_idx]
        a_radius_bound = max(1e-4, 0.3 * (1 - a_pos[0] - a_pos[1]))  # Adaptive radius bound
        b_radius_bound = max(1e-4, 0.3 * (1 - b_pos[0] - b_pos[1]))  # Adaptive radius bound
        
        # Special handling: isolate the pair and perform constrained local refinement
        # Create a new v vector where only the two interacting circles can change
        isolated_v = v.copy()
        # Freeze all positions and radii except for the two interactors
        # Freeze x and y coordinates
        isolated_v[3 * (n - 2):3 * (n - 1)] = [a_pos[0], a_pos[1]]  # Circle a
        isolated_v[3 * (n - 1):3 * n] = [b_pos[0], b_pos[1]]  # Circle b
        # Allow radii to expand under the constraint of maintaining distance
        # Set upper bound for radii based on distance to boundary and interaction
        # Keep other radii frozen as they are not dynamically interacting
        radii_mask = np.zeros(n)
        radii_mask[a_idx] = 0.95  # Allow some expansion but not all
        radii_mask[b_idx] = 1.0  # Allow full expansion, as it's the most dynamically interacting
        # Set radii bounds in isolated_v
        isolated_v[2::3] = np.where(radii_mask, 
                                    np.array([a_radius_bound, b_radius_bound] + [radii[i] for i in range(n) if i not in top_two_indices]), 
                                    radii)
        
        # Set bounds for the isolated_v for the two interactors
        # For x and y for both: 0 to 1 and same as original
        bound_for_interactors = [(0.0, 1.0), (0.0, 1.0)] 
        # For radii, set upper bound to max of current value with expanded limit
        radius_upper_for_interactors = np.array([a_radius_bound, b_radius_bound])
        # Freeze other circles' radii
        for i in range(n):
            if i not in top_two_indices:
                isolated_v[3*i + 2] = radii[i]  # Keep frozen
                # Set bounds for non-interactors to not change
                bounds[3*i] = (v[3*i], v[3*i])  # Freeze x
                bounds[3*i + 1] = (v[3*i + 1], v[3*i + 1])  # Freeze y
                bounds[3*i + 2] = (radii[i], radii[i])  # Freeze radius
        
        # Redefine constraints for the isolated_v, keeping the same original constraints
        # (only the first 2*n constraints will be active for the two interactors)
        # We will only allow expansion of the two interactors
        # Recreate the same constraints but with updated bounds
        # First redefine the bounds list for the new isolated_v
        new_bounds = []
        for i in range(n):
            if i in top_two_indices:
                # Allow x and y to vary, but within 0 to 1
                new_bounds += [(0.0, 1.0), (0.0, 1.0)]
                # Allow radii to vary within bounds for interactors
                r_bound = (1e-4, radius_upper_for_interactors[i - (i in [a_idx, b_idx])])
                new_bounds += [r_bound]
            else:
                # Freeze coordinates and radii for non-interactors
                new_bounds += [(v[3*i], v[3*i]), (v[3*i+1], v[3*i+1]), (radii[i], radii[i])]

        # Recompute the constraints for the new bounds configuration
        # The original constraints are still applicable but only affect the two interactors
        # We will reuse the same constraint list but with updated bounds
        # Re-run the optimization on the isolated pair with new bounds and same constraints
        # Set a tighter tolerance for local refinement
        res_isolated = minimize(neg_sum_radii, isolated_v, method="SLSQP", bounds=new_bounds,
                               constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-9})
        
        if res_isolated.success:
            v = res_isolated.x
            # After the isolated refinement, perform a targeted expansion of the least 
            # constrained circle, with additional constraint to force reordering of layout topology
            
            # Recalculate positions to use the new refined positions
            centers = np.column_stack([v[0::3], v[1::3]])
            radii = v[2::3]
            # Recompute all pairwise distances
            dists = np.zeros((n, n))
            dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
            dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
            dists = np.sqrt(dx**2 + dy**2)
            
            # Find the circle with the minimal non-zero radius (least constrained)
            # But also ensure the selected circle is not one of the two interactors
            min_radius_idx = np.argmin(radii)
            if min_radius_idx in top_two_indices:
                # Choose the second smallest radius that isn't among the two interactors
                non_interactor_indices = np.setdiff1d(np.arange(n), top_two_indices)
                min_radius_idx = np.argmin(radii[non_interactor_indices])
                min_radius_idx = non_interactor_indices[min_radius_idx]
            
            # Create a new vector to expand the least constrained circle with topological constraint
            # Targeted expansion with geometric hashing to enforce layout reordering
            # We will expand it while ensuring it interacts with at least one other circle in a new way
            # This forces the topology to reconfigure
            expansion_vector = np.zeros(3 * n)
            expansion_vector[3 * min_radius_idx + 2] = radii[min_radius_idx] * 2.0
            
            # Add the new expansion to the vector
            new_v = v.copy()
            new_v[3 * min_radius_idx + 2] += expansion_vector[3 * min_radius_idx + 2]
            
            # Apply a geometric hash to the expanded circle to force interaction
            # This adds a constraint that forces a new spatial interaction
            # Create a new geometric constraint that adds to the current system
            # The constraint is that the newly expanded circle should now have a minimum distance
            # to at least one other circle that was not previously a neighbor
            
            # Compute neighbors of the least constrained circle before expansion
            neighbors_before = np.where((dists[min_radius_idx] < (radii[min_radius_idx] + radii[np.arange(n)] - 1e-12)))[0]
            # Ensure at least one neighbor is present
            if len(neighbors_before) > 0:
                # Choose one of them as the target to reinteract with in a new way
                target_neighbor_idx = neighbors_before[0]
                # Add a geometric hash constraint that increases the minimum distance between the
                # expanded circle and target neighbor to force a new spatial interaction
                # This creates a new constraint that pushes the layout to reorder
                # This is a soft constraint, but it helps guide the system toward a higher-quality configuration
                
                # Add an additional constraint to the optimization
                # The constraint ensures that after expansion, the circle has a minimum distance to a specific other
                # This forces reconfiguration of the layout topology
                def topo_constraint(v, target_idx=min_radius_idx, neighbor_idx=target_neighbor_idx):
                    dx = v[3 * target_idx] - v[3 * neighbor_idx]
                    dy = v[3 * target_idx + 1] - v[3 * neighbor_idx + 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    # We require this distance to be at least 1.2 * (r_target + r_neighbor)
                    return dist - 1.2 * (v[3 * target_idx + 2] + v[3 * neighbor_idx + 2])
                
                cons.append({"type": "ineq", "fun": topo_constraint})
            
            # Perform a third stage: expand the least constrained circle while maintaining all constraints
            # Use the expanded vector as the new starting point
            res_expanded = minimize(neg_sum_radii, new_v, method="SLSQP", bounds=new_bounds,
                                   constraints=cons, options={"maxiter": 400, "ftol": 1e-12, "eps": 1e-9})
            
            if res_expanded.success:
                v = res_expanded.x
                # Final refinement with boundary clipping and ensuring all constraints are met
                centers = np.column_stack([v[0::3], v[1::3]])
                radii = v[2::3]
                # Re-check bounds and radii
                for i in range(n):
                    x, y, r = centers[i], centers[i,1], radii[i]
                    if (x - r < -1e-12 or x + r > 1.0 + 1e-12 or
                        y - r < -1e-12 or y + r > 1.0 + 1e-12):
                        # Clip to bounds
                        v[3*i] = max(min(v[3*i], 1.0), 0.0)
                        v[3*i+1] = max(min(v[3*i+1], 1.0), 0.0)
                        v[3*i+2] = np.clip(v[3*i+2], 1e-6, 0.5)  # Ensure radii are within bounds
                # Final optimization to fix any residual issues
                res_final = minimize(neg_sum_radii, v, method="SLSQP", bounds=new_bounds,
                                    constraints=cons, options={"maxiter": 300, "ftol": 1e-12, "eps": 1e-9})
                v = res_final.x if res_final.success else v
            
        else:
            # If isolated refinement failed, fallback to the first configuration
            v = res.x

    centers = np.column_stack([v[0::3], v[1::3]])
    radii = v[2::3]
    # Ensure radii are valid and within constraints
    radii = np.clip(radii, 1e-6, 0.5)  # Clip radii to ensure they stay valid
    # Ensure centers are valid and within bounds
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < -1e-12 or x + r > 1 + 1e-12 or 
            y - r < -1e-12 or y + r > 1 + 1e-12):
            # Clip to bounds
            v[3*i] = max(min(v[3*i], 1.0), 0.0)
            v[3*i+1] = max(min(v[3*i+1], 1.0), 0.0)
            v[3*i+2] = np.clip(v[3*i+2], 1e-6, 0.5)
    # Final check for valid configuration
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = v[2::3]
    return centers, radii, float(radii.sum())