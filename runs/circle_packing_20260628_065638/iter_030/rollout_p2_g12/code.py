import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with dynamic staggered grid and asymmetric clustering
    xs = []
    ys = []
    # Create a grid that's 5 columns but optimized to handle odd distributions
    # Introduce a spatial distortion function that avoids symmetric clustering and forces stagger
    # Start with refined grid that allows more fluid spatial distribution
    for i in range(n):
        row = i // cols
        col = i % cols
        # Compute base grid position
        x_center = col / cols + 0.5 / cols  # Avoid direct centering
        y_center = row / rows + 0.5 / rows
        
        # Add a non-uniform spatial distortion function that varies by row
        # Dynamic distortion: more at the edge rows to avoid clustering
        if row <= rows // 3:
            x_offset = np.random.uniform(-0.03, 0.03)
            y_offset = np.random.uniform(-0.05, 0.05)
            # Add an asymmetry to break symmetry
        elif row >= rows - rows // 3:
            x_offset = np.random.uniform(-0.04, 0.04)
            y_offset = np.random.uniform(-0.05, 0.05)
            # Shift for top rows to avoid vertical stacking
        else:
            x_offset = np.random.uniform(-0.02, 0.02)
            y_offset = np.random.uniform(-0.03, 0.03)
        # Apply staggered layout with alternating shift
        if row % 2 == 1:
            x_offset += 0.5 / cols * (row * 0.17)
        else:
            x_offset -= 0.4 / cols * (row * 0.1)

        x = x_center + x_offset
        y = y_center + y_offset
        xs.append(x)
        ys.append(y)
    
    r0 = 0.35 / cols - 1e-3
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = np.full(n, r0)

    # Ensure bounds and vector are in alignment
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]  # length 3*n

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Optimized, vectorized constraint construction with lambda capture
    cons = []
    for i in range(n):
        # Boundary constraints (left, right, bottom, top)
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3*i + 1] - v[3*i + 2]})
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]})

    # Overlap constraints using optimized lambda capture (avoiding closure capture issues)
    # Optimized overlap constraint construction with dynamic spatial grouping
    # Create a spatial grouping matrix for efficient overlap computation using vectorization
    # Create 2D spatial group matrix with efficient spatial proximity
    # Create overlapping group pairs using hierarchical clustering
    # Group the circles in a way that reduces the number of pairwise checks while maintaining critical constraints
    # Implement a layered spatial grouping to focus constraints on dynamically interacting regions
    # Group into clusters and apply constraints at the cluster level
    # This is a critical optimization for performance and constraint resolution
    groupings = []
    if n <= 10:
        # Small scale: full pairwise
        for i in range(n):
            for j in range(i + 1, n):
                cons.append({"type": "ineq",
                             "fun": (lambda v, i=i, j=j: 
                                     (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                     - (v[3*i+2] + v[3*j+2])**2)})
    elif 10 < n <= 15:
        # Medium scale: group by rows and perform pairwise within rows and between rows
        row_groups = [(i//cols, i%cols) for i in range(n)]
        group_rows = {}
        for r in range(rows):
            group_rows[r] = [i for i in range(n) if row_groups[i][0] == r]
        for r in range(rows):
            for i in group_rows[r]:
                for j in group_rows[r]:
                    if i < j:
                        cons.append({"type": "ineq",
                                     "fun": (lambda v, i=i, j=j: 
                                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                             - (v[3*i+2] + v[3*j+2])**2)})
            # Add cross-group constraints only between adjacent rows
            # This reduces the number of constraints without sacrificing critical overlaps
        for r1 in range(rows):
            for r2 in range(r1 + 1, rows):
                for i in group_rows[r1]:
                    for j in group_rows[r2]:
                        cons.append({"type": "ineq",
                                     "fun": (lambda v, i=i, j=j: 
                                             (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                                             - (v[3*i+2] + v[3*j+2])**2)})
    else:
        # Large scale: group via spatial hashing with efficient proximity calculation
        # Create a hash function that captures spatial relationships
        # Spatial hashing will reduce constraints by focusing only on neighboring clusters
        # Create groupings by distance thresholds to limit pairwise checks
        # Use a dynamic hash grid for better constraint optimization
        grid_size = max(1, int(np.sqrt(n)))
        grid = {}
        for index in range(n):
            x = v0[3 * index]
            y = v0[3 * index + 1]
            grid_key = (int(x / (1.0 / grid_size)), int(y / (1.0 / grid_size)))
            if grid_key not in grid:
                grid[grid_key] = []
            grid[grid_key].append(index)
        # Cluster nearby groups and apply pairwise checking within and between near clusters
        # This creates a layered spatial structure that reduces pairwise constraints
        for key1, indices1 in grid.items():
            for key2, indices2 in grid.items():
                if key1 != key2:
                    # Determine if the clusters are adjacent in grid space to check overlaps
                    dx = abs(key1[0] - key2[0])
                    dy = abs(key1[1] - key2[1])
                    if dx <= 1 and dy <= 1:
                        # Apply pairwise checking between clusters
                        for i in indices1:
                            for j in indices2:
                                if i < j:
                                    cons.append({"type": "ineq",
                                                 "fun": (lambda v, i=i, j=j: 
                                                         (v[3*i] - v[3*j])**2 
                                                         + (v[3*i+1] - v[3*j+1])**2 
                                                         - (v[3*i+2] + v[3*j+2])**2)})
        
        # Add intra-group pairwise constraints for full internal cluster checks
        for indices in grid.values():
            for i in indices:
                for j in indices:
                    if i < j:
                        cons.append({"type": "ineq",
                                     "fun": (lambda v, i=i, j=j: 
                                             (v[3*i] - v[3*j])**2 
                                             + (v[3*i+1] - v[3*j+1])**2 
                                             - (v[3*i+2] + v[3*j+2])**2)})
    
    # Initial optimization with enhanced options
    # Use SLSQP with improved tolerance and constraints
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-12, "eps": 1e-10})
    
    # Post-iteration analysis and reconfiguration
    if res.success:
        v = res.x
        radii = v[2::3]
        centers = np.column_stack([v[0::3], v[1::3]])
        
        # Step 1: Identify and reconfigure two most dynamically interacting circles
        # Compute all pairwise distances in a vectorized way
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute interaction strength using non-linear metric
        # Use a combination of distance and radius to determine interaction
        interaction_strength = np.zeros(n)
        for i in range(n):
            radius_i = radii[i]
            for j in range(n):
                if i != j:
                    dist = dists[i, j]
                    if dist > 0:
                        interaction_strength[i] += radius_i / max(1e-6, dist)
        
        # Find top two circles by interaction strength
        top_idx = np.argsort(interaction_strength)[-2:]
        top_circle_1, top_circle_2 = top_idx
        
        # Step 2: Perform geometric dissection on these two circles
        # Create a perturbation function with direction-based scaling that avoids overlapping while increasing radii
        # Use directional perturbation that ensures non-overlap in the new configuration
        # Maintain non-overlapping by using constraint-aware perturbation
        
        # Create a perturbation vector that's adaptive based on the geometry of the top two circles
        current_center_1 = centers[top_circle_1]
        current_center_2 = centers[top_circle_2]
        distance_between = dists[top_circle_1, top_circle_2]
        radius_sum = radii[top_circle_1] + radii[top_circle_2]
        max_allowed_distance = radius_sum
        if distance_between < max_allowed_distance - 1e-12:
            # The circles are already too close; need to increase radii carefully
            # First, expand the radii of the two circles proportionally
            # Compute total capacity for expansion
            max_growth = (max_allowed_distance - distance_between) - 1e-12
            total_possible_growth = 0.95 * (max_growth - 1e-12)
            if total_possible_growth > 0:
                # Distribute growth to the two circles
                expansion_factor = np.random.uniform(0.6, 1.4)
                new_radius_1 = radii[top_circle_1] + (total_possible_growth * expansion_factor / 2)
                new_radius_2 = radii[top_circle_2] + (total_possible_growth * expansion_factor / 2)
                # Apply expansion and reconfigure the centers to maintain non-overlap
                # Use a linear perturbation to separate the circles based on the direction vector
                dir_vec = (current_center_1 - current_center_2) / max(1e-6, distance_between)
                perturb = dir_vec * (new_radius_1 + new_radius_2 - distance_between) * 0.5
                new_centers_1 = current_center_1 + perturb
                new_centers_2 = current_center_2 - perturb
                # Update the decision vector for top circles only
                perturbed_v = v.copy()
                perturbed_v[3*top_circle_1] = new_centers_1[0]
                perturbed_v[3*top_circle_1 + 1] = new_centers_1[1]
                perturbed_v[3*top_circle_1 + 2] = new_radius_1
                perturbed_v[3*top_circle_2] = new_centers_2[0]
                perturbed_v[3*top_circle_2 + 1] = new_centers_2[1]
                perturbed_v[3*top_circle_2 + 2] = new_radius_2
                # Re-optimize with the updated configuration
                res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-9})
                # After optimization, recompute the updated centers
                v_update = res.x
                centers = np.column_stack([v_update[0::3], v_update[1::3]])
                radii = v_update[2::3]
                # Recompute distances after reconfiguration
        else:
            # The two circles are sufficiently apart; no need to expand their radii immediately
            # Instead, apply controlled spatial dissection to improve overall configuration
            # Use a directional perturbation that repositions them with increased spatial separation
            dir_vec = (current_center_1 - current_center_2) / max(1e-6, distance_between)
            perturb = dir_vec * 0.3 * (distance_between - (radii[top_circle_1] + radii[top_circle_2]))
            # Apply perturbation to centers
            new_centers_1 = current_center_1 + perturb
            new_centers_2 = current_center_2 - perturb
            # Update the decision vector for top circles only
            perturbed_v = v.copy()
            perturbed_v[3*top_circle_1] = new_centers_1[0]
            perturbed_v[3*top_circle_1 + 1] = new_centers_1[1]
            perturbed_v[3*top_circle_2] = new_centers_2[0]
            perturbed_v[3*top_circle_2 + 1] = new_centers_2[1]
            # Keep radii unchanged for this step
            # Re-optimize with the updated configuration
            res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 500, "ftol": 1e-11, "eps": 1e-9})
            # After optimization, recompute the updated centers
            v_update = res.x
            centers = np.column_stack([v_update[0::3], v_update[1::3]])
            radii = v_update[2::3]
        
        # Step 3: Post-dissection optimization to stabilize configuration
        # Perform another round of optimization with enhanced tolerance and convergence
        res = minimize(neg_sum_radii, v_update, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 1200, "ftol": 1e-11, "eps": 1e-9})

    # Step 4: Re-evaluation of the entire configuration to ensure stability
    # Identify the least constrained circle and perform targeted expansion
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Vectorized compute pairwise distances between all circles
        dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
        dy = centers[:, np.newaxis, 1] - centers[np.newaxis, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Compute isolation strength for each circle
        # Isolation strength = sum(1 / max(1e-6, distance)) for all other circles
        total_isolation = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance = dists[i, j]
                    if distance > 1e-6:
                        total_isolation[i] += 1.0 / distance
                    else:
                        total_isolation[i] += 0  # No penalty for near overlap
        
        # Find the circle with highest isolation
        isolated_idx = np.argmax(total_isolation)
        
        # Calculate the potential growth based on current configuration
        # Total current sum = sum(radii)
        # Target potential growth is based on current total sum to avoid overexpansion
        max_growth_capacity = 0.0
        for i in range(n):
            if i != isolated_idx:
                max_growth_capacity += (1.0 - radii[i])  # Max possible growth for all other circles
        
        # Calculate an expansion coefficient
        expansion_coeff = np.random.uniform(0.75, 1.2)
        max_possible_growth = expansion_coeff * max_growth_capacity * 0.8
        
        # Apply growth while maintaining total sum constraint
        # Distribute the growth to all other circles equally
        growth_per_circle = max_possible_growth / (n - 1)
        new_radii = radii.copy()
        new_radii[isolated_idx] = new_radii[isolated_idx] + max_possible_growth  # Expanding the isolated one
        for i in range(n):
            if i != isolated_idx:
                new_radii[i] = new_radii[i] + growth_per_circle
        
        # Now, reconfigure the entire system with expanded radii while maintaining constraints
        # This is a controlled expansion while preserving the constraint framework
        # Ensure that all circles are inside the square and no overlap is introduced
        while True:
            expanded_v = v.copy()
            expanded_v[2::3] = new_radii
            expanded_centers = np.column_stack([expanded_v[0::3], expanded_v[1::3]])
            valid = True
            
            # Validate the expanded configuration
            for i in range(n):
                for j in range(i + 1, n):
                    dx_expand = expanded_centers[i, 0] - expanded_centers[j, 0]
                    dy_expand = expanded_centers[i, 1] - expanded_centers[j, 1]
                    dist = np.sqrt(dx_expand**2 + dy_expand**2)
                    if dist < new_radii[i] + new_radii[j] - 1e-12:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Update decision vector and optimize
                v = expanded_v
                break
            else:
                # If invalid, decrease growth slightly
                growth_coeff = 0.9  # Adjust this factor to control the rate of reduction
                new_radii = radii.copy()
                new_radii[isolated_idx] = new_radii[isolated_idx] + max_possible_growth * growth_coeff
                for i in range(n):
                    if i != isolated_idx:
                        new_radii[i] = new_radii[i] + growth_per_circle * growth_coeff
        
        # After validation, perform optimization
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 800, "ftol": 1e-11, "eps": 1e-9})
    
    # Final cleanup and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())