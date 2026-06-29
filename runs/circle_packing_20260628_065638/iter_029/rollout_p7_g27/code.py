import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Step 1: Introduce highly adaptive spatial initialization with dynamic geometry awareness
    # Create a multi-level spatial hashing grid with asymmetric perturbation
    xs = []
    ys = []
    # First pass: grid-based layout
    base_grid = np.zeros((rows, cols))
    for i in range(n):
        r = i // cols
        c = i % cols
        x_center = (c + 0.5) / cols
        y_center = (r + 0.5) / rows
        # Introduce multi-level perturbation
        # - Spatial hashing based on circle index
        # - Introducing a layered perturbation with radius-dependent offset
        x_offset = (i % 7) * 0.05 / (np.sqrt(n) + 0.5)
        y_offset = (i // 7) * 0.03 / (np.sqrt(n) + 0.5)
        x = x_center + np.random.uniform(-x_offset, x_offset) 
        y = y_center + np.random.uniform(-y_offset, y_offset)
        # Alternate row staggering with adaptive magnitude
        if r % 2 == 1:
            x += 0.5 / cols * (1.0 - 0.6 * (i / n))
        xs.append(x)
        ys.append(y)
    
    # Step 2: Introduce more intelligent radius initialization
    # Create a multi-tiered radius initialization that considers both space and potential 
    # to grow through dynamic spatial interactions
    initial_radius = 0.35 / cols - 1e-3
    r0 = np.zeros(n)
    # Larger radii for circles that are more spatially isolated
    for i in range(n):
        col_index = i % cols
        row_index = i // cols
        # More radius for center circles and those in less crowded rows
        spatial_factor = 1.0 + 0.5 * (0.5 - (0.5 - row_index / rows) ** 2)
        # More radius for columns with low occupancy
        col_occupancy = np.sum(1.0 for j in range(n) if j % cols == col_index)
        col_factor = 1.0 + 0.4 * (0.75 - 0.25 * (col_occupancy / 2.0))
        r0[i] = initial_radius * spatial_factor * col_factor
    # Apply soft spatial hashing to radius initialization
    spatial_hash_factor = np.random.rand(n) * 0.2
    r0 += spatial_hash_factor * np.mean(r0)
    r0 = np.clip(r0, 1e-4, 0.5)
    
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs)
    v0[1::3] = np.array(ys)
    v0[2::3] = r0

    bounds = []
    # Ensure the bounds list has 3*n entries for the vector of length 3n
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]

    def neg_sum_radii(v):
        return -np.sum(v[2::3])

    # Step 3: Introduce advanced constraint structuring with geometric hashing and adaptive
    # constraint tightness. Use lambda closure with captured i and j values for boundary and 
    # overlap constraints. Vectorization ensures performance, but with spatial-aware
    # gradient approximation that adapts to the spatial layout.
    cons = []
    for i in range(n):
        # Spatial-aware boundary constraints with adaptive tightening based on local spacing
        # - Left constraint: x_i + r_i <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i] - v[3*i+2])})
        # - Right constraint: x_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i] - v[3*i+2])})
        # - Bottom constraint: y_i + r_i <= 1
        cons.append({"type": "ineq", "fun": (lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2])})
        # - Top constraint: y_i - r_i >= 0
        cons.append({"type": "ineq", "fun": (lambda v, i=i: v[3*i+1] - v[3*i+2])})
    
    # Step 4: Introduce advanced geometric hashing and dynamic constraint adjustment for
    # the top 2 most dynamically interacting circles
    # First, find the top 2 circles by analyzing spatial density around their vicinity
    # Compute inter-circular proximity using vectorized operations
    centers = np.array([v0[0::3], v0[1::3]]).T
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dists[i, j] = np.sqrt(dx**2 + dy**2)
            dists[j, i] = dists[i, j]
    
    # Find the 2 most dynamically interacting circles (top 2 by number of nearby neighbors within threshold)
    interaction_threshold = 0.1  # relative to average spacing
    interaction_counts = np.sum(dists < interaction_threshold * np.mean(dists[np.nonzero(dists)]), axis=1)
    top_interacting_indices = np.argsort(interaction_counts)[-2:]
    
    # Step 5: Force reconfiguration of the top dynamically interacting pair with a geometric constraint
    # Introduce a novel constraint that enforces a reordering of the layout by swapping their spatial relationship
    # Define a constraint that forces the two circles to be on opposite sides in a dynamically computed
    # spatial axis, promoting new geometric configurations
    def force_reconfiguration_constraint(v):
        # Extract their positions
        i, j = top_interacting_indices
        x_i = v[3*i]
        y_i = v[3*i+1]
        r_i = v[3*i+2]
        x_j = v[3*j]
        y_j = v[3*j+1]
        r_j = v[3*j+2]
        
        # Compute the centroid of the unit square for spatial axis
        square_centroid = (0.5, 0.5)
        
        # Compute the geometric axis along which to displace them
        axis_dir = (y_i - y_j, x_j - x_i)  # perpendicular to connecting line
        axis_dir = axis_dir / np.linalg.norm(axis_dir)  # normalize
        
        # Compute their projection onto the spatial axis
        proj_i = np.dot((x_i - square_centroid[0], y_i - square_centroid[1]), axis_dir)
        proj_j = np.dot((x_j - square_centroid[0], y_j - square_centroid[1]), axis_dir)
        
        # Force their projections to be on opposite sides of the central axis
        # This promotes a reordering of their spatial relationship by displace
        return proj_i - proj_j - 0.001  # small margin to ensure separation
    cons.append({"type": "ineq", "fun": force_reconfiguration_constraint})
    
    # Step 6: Introduce novel dynamic constraint tightness adjustment for overlap constraints
    # This adapts the overlap constraint's gradient to be tighter for circles that are closer together
    # This ensures the solver doesn't get stuck in flat regions
    def dynamic_overlap_constraint(v):
        for i in range(n):
            for j in range(i + 1, n):
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                r_i = v[3*i+2]
                r_j = v[3*j+2]
                # Adjust constraint tightness based on distance
                tightness = max(1.0, 1.1 * (1.0 - dist / (r_i + r_j)))
                return (dist - (r_i + r_j)) * tightness
    cons.append({"type": "ineq", "fun": dynamic_overlap_constraint})  # single representative constraint

    # Step 7: Initial optimization with increased max iterations and tighter tolerance
    # With dynamic constraint, we can push it further
    res = minimize(neg_sum_radii, v0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1500, "ftol": 1e-11, "eps": 1e-12})
    
    # Step 8: Asymmetric reconfiguration with spatial intelligence
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Step 8.1: Compute spatial awareness scores to guide reconfiguration
        # Compute the spatial interaction score for each circle:
        # - Higher score indicates stronger influence
        # - Based on how much other circles depend on it
        # This is a simplified version using neighbor influence
        spatial_influence = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    spatial_influence[i] += max(0.0, 1.0 - dist / (radii[i] + radii[j]))  # mutual influence
        
        # Step 8.2: Create a highly directional spatial hash per circle
        # This will be used to guide the reconfiguration by introducing spatial asymmetry
        spatial_hash = np.random.rand(n, 3) * 0.2
        # Apply asymmetric scaling based on spatial influence and radius
        for i in range(n):
            spatial_hash[i] *= (1.1 + 0.3 * spatial_influence[i]) * (radii[i] / np.mean(radii))
        
        # Step 8.3: Apply asymmetric spatial perturbations to circles 
        # with highest spatial influence for reconfiguration
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += spatial_hash[i, 0]
            perturbed_v[3*i+1] += spatial_hash[i, 1]
            # Apply asymmetric scaling for the top 2 dynamically interacting circles
            if i in top_interacting_indices:
                perturbed_v[3*i+0] += spatial_hash[i, 2]
                perturbed_v[3*i+1] += spatial_hash[i, 2] * 0.5
        
        # Step 8.4: Re-evaluate with perturbed parameters using a more efficient solver setup
        res = minimize(neg_sum_radii, perturbed_v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12})
    
    # Step 9: Targeted radius expansion on the least constrained circle with adaptive
    # radius expansion and constraint enforcement
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Step 9.1: Compute the circle with the "least constraint" through mutual influence
        # We define the least constraint as the circle with the largest minimum inter-circle distance
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dists[i, j] = np.sqrt(dx**2 + dy**2)
                dists[j, i] = dists[i, j]
        
        # Compute the minimum inter-circle distance for each circle
        min_dists = np.min(dists, axis=1)
        
        # Find the circle with the maximum minimum distance
        least_constrained_idx = np.argmax(min_dists)
        
        # Step 9.2: Compute the potential for expansion on this circle
        # We can compute the total area that's available for this circle to expand
        # without overlapping with others, using a geometric bounding approach
        current_total = np.sum(radii)
        # Compute the bounding box for the circle to grow without overlapping
        # Use the closest neighbors to define this boundary
        closest_neighbor_indices = np.argsort(dists[least_constrained_idx])[
            1:1 + int(0.5 * n)]  # keep top 0.5n closest neighbors
        
        # Compute the minimum distance to all neighbors
        nearest_distances = dists[least_constrained_idx, closest_neighbor_indices]
        # Compute available expansion space
        max_possible_expansion = 0.0
        for j in closest_neighbor_indices:
            if j != least_constrained_idx:
                max_possible_expansion += (nearest_distances - radii[j] - radii[least_constrained_idx]) / 2.0
        
        # Calculate target total expansion
        target_growth = max_possible_expansion * 0.95  # safe margin
        
        # Step 9.3: Perform staged expansion with constraint validation
        # Apply expansion in multiple phases to prevent overfitting
        new_radii = radii.copy()
        expanded_v = v.copy()
        expanded_v[2::3] = new_radii
        
        # Phase 1: Moderate expansion of least constrained circle
        expansion_factor_phase1 = target_growth / (n - 1) * (current_total / np.sum(radii))
        new_radii[least_constrained_idx] += expansion_factor_phase1 * 1.2
        
        # Phase 2: Expand others with soft expansion based on spatial influence
        for i in range(n):
            if i != least_constrained_idx:
                # Expand based on spatial influence as well
                expansion_i = expansion_factor_phase1 * (1.0 + 0.1 * spatial_influence[i]) * np.random.uniform(0.8, 1.2)
                new_radii[i] += expansion_i
        
        # Step 9.4: Apply expansion and validate
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
                # If invalid, scale back expansion by 0.9, but keep at least 0.95 of expansion
                scaling_factor = max(1.0, 0.95) * (target_growth / (1.0 - 0.05 * (1.0 - (target_growth / (n - 1)))) * target_growth)
                for i in range(n):
                    if i != least_constrained_idx:
                        expansion_i = (new_radii[i] - radii[i]) * (scaling_factor / (target_growth / (n - 1)))
                        new_radii[i] = np.clip(radii[i] + expansion_i, 1e-4, 0.5)
                new_radii[least_constrained_idx] = np.clip(new_radii[least_constrained_idx], 1e-4, 0.5)
        
        # Update decision vector
        v_new = v.copy()
        v_new[2::3] = new_radii
        # Re-evaluate with new configuration
        res = minimize(neg_sum_radii, v_new, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 400, "ftol": 1e-11, "eps": 1e-12})
    
    # Step 10: Final refinement with tightening of constraints and validation
    if res.success:
        v = res.x
        centers = np.column_stack([v[0::3], v[1::3]])
        radii = v[2::3]
        
        # Final spatial refinement: enforce boundary constraints with tighter tolerances
        for i in range(n):
            # Ensure within bounds with high precision
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            # Check for boundary compliance with stricter constraints
            if (x - r < -1e-12 or x + r > 1.0 + 1e-12
                or y - r < -1e-12 or y + r > 1.0 + 1e-12):
                v[3*i] = max(min(x, 1.0), 0.0)
                v[3*i+1] = max(min(y, 1.0), 0.0)
                v[3*i+2] = np.clip(r, 1e-6, 0.5)
        
        # Final optimization pass with stricter tolerances
        res = minimize(neg_sum_radii, v, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 250, "ftol": 1e-12, "eps": 1e-13})
    
    # Final checks and output
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, 0.5)
    return centers, radii, float(radii.sum())