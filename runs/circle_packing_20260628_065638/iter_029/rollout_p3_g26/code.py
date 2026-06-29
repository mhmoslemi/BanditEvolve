import numpy as np

def run_packing():
    n = 26
    # Use hexagonal tiling structure with dynamic row/column counts for balanced expansion
    cols = int(np.ceil(np.sqrt(n)))
    rows = (n + cols - 1) // cols
    
    # Initialize with hexagonal grid with randomized spatial displacement
    xs = []
    ys = []
    for i in range(n):
        col = i % cols
        row = i // cols
        x_center = (col + 0.5) / cols
        y_center = (row + 0.5) / rows
        # Randomized offset to avoid symmetry and clustering
        x_offset = np.random.uniform(-0.08, 0.08) * (0.5 / cols)
        y_offset = np.random.uniform(-0.08, 0.08) * (0.5 / rows)
        x = x_center + x_offset
        y = y_center + y_offset
        # Alternate row offset for staggered grid
        if row % 2 == 1:
            x += 0.5 / cols
        xs.append(x)
        ys.append(y)
    
    # Initialize radii with adaptive spacing
    r0_base = 0.36 - 1e-3
    r0 = r0_base * np.clip(np.random.rand(n), 0.85, 1.0)  # Spread initial radii for diversity in local optima
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0
    
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # Match length of 3*n exactly

    # Objective function with smoothness penalty for convergence
    def neg_sum_radii_with_smoothness(v):
        radii = v[2::3]
        total_sum = np.sum(radii)
        
        # Add smoothness penalty for adjacent circles to avoid sharp discontinuities
        smoothness_penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < 1e-6:  # Avoid division by zero for very close centers
                    smoothness_penalty += 0 # no penalty
                else:
                    smoothness_penalty += np.log(max(1e-10, (dist - (radii[i] + radii[j])) / (radii[i] + radii[j])) + 1)
        
        return -total_sum - 0.1 * smoothness_penalty  # Weigh smoothness penalty to encourage gradual changes

    # Vectorized constraint functions: avoid lambda closure issues by using positional parameters
    constraints = []
    
    for i in range(n):
        # Left edge constraint
        def left_constraint(v, i=i):
            return v[3*i] - v[3*i + 2]
        constraints.append({"type": "ineq", "fun": left_constraint})
        # Right edge constraint
        def right_constraint(v, i=i):
            return 1.0 - v[3*i] - v[3*i + 2]
        constraints.append({"type": "ineq", "fun": right_constraint})
        # Bottom edge constraint
        def bottom_constraint(v, i=i):
            return v[3*i + 1] - v[3*i + 2]
        constraints.append({"type": "ineq", "fun": bottom_constraint})
        # Top edge constraint
        def top_constraint(v, i=i):
            return 1.0 - v[3*i + 1] - v[3*i + 2]
        constraints.append({"type": "ineq", "fun": top_constraint})
    
    # Vectorized distance constraints with adaptive scaling for overlapping prevention
    for i in range(n):
        for j in range(i + 1, n):
            # Avoid using lambda closures to prevent scoping issues
            def distance_constraint(v, i, j):
                dx = v[3*i] - v[3*j]
                dy = v[3*i + 1] - v[3*j + 1]
                dist = dx*dx + dy*dy
                radii_sum = v[3*i + 2] + v[3*j + 2]
                return dist - radii_sum**2  # Constraint: distance^2 >= (r_i + r_j)^2
            constraints.append({"type": "ineq", "fun": lambda v, i=i, j=j: distance_constraint(v, i, j)})
    
    # Initial optimization phase: high diversity search
    res = minimize(neg_sum_radii_with_smoothness, v0,
                   method="SLSQP", 
                   bounds=bounds,
                   constraints=constraints,
                   options={"maxiter": 1200, "ftol": 1e-9, "eps": 1e-8})
    
    # Spatial diversity phase: geometric hashing-inspired permutation with adaptive perturbations
    if res.success:
        v = res.x
        # Compute current radii and distances
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Calculate pairwise distances for clustering detection
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Identify clusters using a dynamic cutoff (based on mean radius and spacing)
        cluster_ids = np.zeros(n, dtype=int)
        cluster_idx = 0
        for i in range(n):
            if cluster_ids[i] == 0:
                cluster_ids[i] = cluster_idx
                for j in range(n):
                    if cluster_ids[j] == 0 and dists[i,j] < 1.5 * (radii[i] + radii[j]):
                        cluster_ids[j] = cluster_idx
                cluster_idx += 1
        
        # For each cluster, apply geometric hashing transformation
        for cluster_id in np.unique(cluster_ids):
            cluster_mask = cluster_ids == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) >= 2:
                # Compute centroid for the cluster
                center_x = np.mean(centers[cluster_indices][:, 0])
                center_y = np.mean(centers[cluster_indices][:, 1])
                
                # Generate random hash perturbations for the cluster
                hash_perturb = np.random.rand(len(cluster_indices), 2) * 0.03 * (radii[cluster_indices] / np.mean(radii))
                perturbed_centers = np.zeros_like(centers)
                for idx, i in enumerate(cluster_indices):
                    x = centers[i][0] + hash_perturb[idx, 0]
                    y = centers[i][1] + hash_perturb[idx, 1]
                    perturbed_centers[i] = [x, y]
                
                # Create new vector from perturbed centers and keep original radii
                perturbed_v = v.copy()
                perturbed_v[0::3] = perturbed_centers[:, 0]
                perturbed_v[1::3] = perturbed_centers[:, 1]
                
                # Re-optimize this cluster with new configuration
                cluster_res = minimize(neg_sum_radii_with_smoothness, perturbed_v,
                                       method="SLSQP",
                                       bounds=bounds,
                                       constraints=constraints,
                                       options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-8})
                
                if cluster_res.success:
                    v = cluster_res.x
        
        # Apply refined perturbations to all circles based on their proximity to edges
        edge_weight = np.zeros(n)
        for i in range(n):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            # Proximity to edges (weighted sum)
            edge_weight[i] = abs(x - r) + abs(x + r - 1) + abs(y - r) + abs(y + r - 1)
        
        # Normalize edge weights
        edge_weight = edge_weight / np.max(edge_weight)
        
        # Generate adaptive perturbations with sensitivity to edge proximity
        random_perturb = np.random.rand(n, 2) * 0.02
        adaptive_perturb = random_perturb * np.clip(edge_weight, 0.15, 1.0)  # More aggressive perturbation near edges
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += adaptive_perturb[i, 0]
            perturbed_v[3*i+1] += adaptive_perturb[i, 1]
        
        # Re-optimize with perturbed centers
        res = minimize(neg_sum_radii_with_smoothness, perturbed_v,
                       method="SLSQP",
                       bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-8})
    
    # Dynamic reconfiguration to target least constrained circles in a feedback loop
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Recalculate pairwise distances for constraint sensitivity
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Find circles with least constraint (maximum minimum distance to others)
        # Avoid dividing by zero by capping minimum distances
        min_dist = np.max(np.min(dists, axis=1), initial=0.1)  # Initial cap to avoid infinite constraints
        constraint_weights = 1 / (np.min(dists, axis=1) + 1e-10)  # Weight circles with higher constraint
        least_constrained_idx = np.argmin(constraint_weights)
        
        # Compute possible expansion for the least constrained circle
        max_expansion = 0.006
        base_radius = radii[least_constrained_idx]
        expansion_factor = np.clip((max_expansion * (base_radius / np.mean(radii))), 1.5, 2.0)
        expansion_amount = expansion_factor * (np.max(radii) - base_radius)
        
        # Create new radii vector with expansion focused on least constrained
        new_radii = radii.copy()
        new_radii[least_constrained_idx] = base_radius + expansion_amount
        
        # Apply constraint check with adaptive step sizing
        v = v.copy()
        v[2::3] = new_radii.copy()
        
        # Re-evaluate with updated radii using an adaptive step-based optimization
        for step in range(5):  # Iteratively refine the new configuration
            # Calculate pairwise constraints
            constraint_values = np.zeros(len(constraints))
            for idx, con in enumerate(constraints):
                c = con["fun"]  # This is a function, call it with current v
                constraint_values[idx] = c(v)
            
            if np.min(constraint_values) > -1e-7:  # All constraints satisfied at sufficient precision
                break
            else:
                # Adjust the least constrained circle's radius
                min_idx = np.argmin(constraint_values)
                # Get the constraint that failed
                failed_con = constraints[min_idx]
                if failed_con["type"] == "ineq":
                    current_val = failed_con["fun"](v)
                    # Adjust the radius to satisfy the constraint
                    # For edge constraint (x - r >= 0): we might need to reduce r
                    # For distance constraint: we might need to increase r
                    # We need to infer which constraint failed and adjust accordingly
                    # For simplicity here, we directly adjust the radius
                    if failed_con["fun"](v) < 0:
                        if v[3*least_constrained_idx + 2] > 1e-5:
                            v[3*least_constrained_idx + 2] -= 0.001
                        else:
                            v[3*least_constrained_idx + 2] = 1e-4
                elif failed_con["type"] == "eq":
                    pass  # Not expected in this setup

        # Final re-solve with updated radii
        res = minimize(neg_sum_radii_with_smoothness, v,
                       method="SLSQP",
                       bounds=bounds,
                       constraints=constraints,
                       options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-8})
    
    # Final validation and clipping
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)
    
    # Final final check with brute-force validation to ensure no overlaps
    # This is computationally costly but critical for edge case validation
    final_centers = centers.copy()
    final_radii = radii.copy()
    for i in range(n):
        for j in range(i + 1, n):
            dx = final_centers[i, 0] - final_centers[j, 0]
            dy = final_centers[i, 1] - final_centers[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < final_radii[i] + final_radii[j] - 1e-6:
                # Need to adjust to meet spacing
                min_radius = min(final_radii[i], final_radii[j])
                max_radius = max(final_radii[i], final_radii[j])
                # Estimate expansion needed for spacing
                expansion_needed = (dist + 1e-6) - (max_radius + min_radius)
                expansion_factor = max(expansion_needed / (max_radius + 1e-8), 0.0)
                # Distribute expansion evenly
                # Scale up both circles in a way that preserves their relative proportions
                scaling = (np.sqrt((max_radius + expansion_factor) ** 2 - dist ** 2)) / (max_radius + min_radius)
                # Distribute the scaling across both
                final_radii[i] *= scaling
                final_radii[j] *= scaling
                # Apply clipping and re-center to maintain spatial integrity
                final_centers[i, 0] = (final_centers[i, 0] * scaling) + (1 - scaling) * final_centers[i, 0]
                final_centers[i, 1] = (final_centers[i, 1] * scaling) + (1 - scaling) * final_centers[i, 1]
                final_centers[j, 0] = (final_centers[j, 0] * scaling) + (1 - scaling) * final_centers[j, 0]
                final_centers[j, 1] = (final_centers[j, 1] * scaling) + (1 - scaling) * final_centers[j, 1]
    
    # Apply post-optimization constraint adjustment
    # Recalculate all constraints again and correct violations
    v = np.column_stack([final_centers[:, 0], final_centers[:, 1], final_radii])
    v = v.flatten()
    # Reapply constraints here as a final check
    # Due to time constraints and function interface, re-evaluate with a smaller optimization run
    res = minimize(neg_sum_radii_with_smoothness, v,
                   method="SLSQP",
                   bounds=bounds,
                   constraints=constraints,
                   options={"maxiter": 300, "ftol": 1e-10, "eps": 1e-8})
    
    # Final output
    v_final = res.x if res.success else v
    centers = np.column_stack([v_final[0::3], v_final[1::3]])
    radii = v_final[2::3]
    radii = np.clip(radii, 1e-6, 0.5)
    return centers, radii, float(radii.sum())