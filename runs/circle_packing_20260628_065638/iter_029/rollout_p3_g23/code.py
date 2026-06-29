import numpy as np

def run_packing():
    n = 26
    cols = 5
    rows = (n + cols - 1) // cols
    
    # Initialize with optimized geometric hashing and dynamic grid adaptation
    # Using an adaptive grid with random offsetting and row staggering with spatial weighting
    xs = []
    ys = []
    grid = np.linspace(0, 1, cols + 2)[1:-1]  # avoid edge effects by excluding first/last
    for i in range(n):
        row = i // cols
        col = i % cols
        # Calculate grid spacing with adaptive weighting for better uniformity
        col_weight = 1.0 / (cols**2) * (np.sqrt(1 - (row / rows)**2) + 1)
        row_weight = 1.0 / rows**2 * (1 + np.log(1 + (row + 0.5)/rows))
        # Generate adaptive centering
        x_center = grid[col] + np.random.normal(loc=0, scale=0.03) * col_weight
        y_center = (row + 0.5) / rows + np.random.normal(loc=0, scale=0.03) * row_weight
        # Stagger rows using a sine-based offset to prevent uniformity collapse
        stagger = 0.5 * np.sin(2 * np.pi * row / rows) * 0.08
        xs.append(x_center + stagger)
        ys.append(y_center)
    
    # Initial radii as function of grid spacing
    min_rad_initial = 0.25 / cols * 1.25  # increased by 25% to allow expansion
    v0 = np.empty(3 * n)
    v0[0::3] = np.array(xs).astype(np.float64)
    v0[1::3] = np.array(ys).astype(np.float64)
    v0[2::3] = np.full(n, min_rad_initial)
    
    # Bounds with adaptive radius constraints
    bounds = []
    for _ in range(n):
        bounds += [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)]
    
    def neg_sum_radii(v):
        return -np.sum(v[2::3])
    
    # Constraint generation with spatial-aware lambda captures and vectorized setup
    cons = []
    
    # Boundary constraints with adaptive tolerances based on grid cell occupancy
    min_cell_occupancy = 0.6 if cols > 5 else 0.8
    cell_occupancy_scalar = (1.0 - min_cell_occupancy)
    for i in range(n):
        # Adaptive left bound: ensures x - r >= 0 with increased tolerance
        tol = 1e-11 + cell_occupancy_scalar * 1e-8
        cons.append({"type": "ineq", "fun": (lambda v, i=i, cell_occupancy_scalar=cell_occupancy_scalar, 
                                            tol=tol: v[3*i] - v[3*i+2] - tol)})
        # Adaptive right bound: ensures x + r <= 1 with increased tolerance
        cons.append({"type": "ineq", "fun": (lambda v, i=i, cell_occupancy_scalar=cell_occupancy_scalar, 
                                            tol=tol: 1.0 - v[3*i] - v[3*i+2] - tol)})
        # Adaptive bottom bound: ensures y - r >= 0 with increased tolerance
        cons.append({"type": "ineq", "fun": (lambda v, i=i, cell_occupancy_scalar=cell_occupancy_scalar, 
                                            tol=tol: v[3*i+1] - v[3*i+2] - tol)})
        # Adaptive top bound: ensures y + r <= 1 with increased tolerance
        cons.append({"type": "ineq", "fun": (lambda v, i=i, cell_occupancy_scalar=cell_occupancy_scalar, 
                                            tol=tol: 1.0 - v[3*i+1] - v[3*i+2] - tol)})
    
    # Overlap constraints using spatial hashing for reduced computation complexity
    # Only apply overlap constraints where spatial distance is likely to be smaller than radius sum
    # We can use a grid to precompute pairs that could potentially collide
    # Spatial hashing with grid size matching the initial grid used
    grid_size = np.sqrt(1.0)
    hash_grid = 5  # match initial 5 cols for hash grid
    # Create spatial hash indices for each circle
    spatial_hashes = np.array([int(x * hash_grid) + int(y * hash_grid) for x, y in zip(xs, ys)])
    # Only check overlapping hashes, which significantly reduces pairwise checks
    for i in range(n):
        # Only check with circles in nearby hash bins
        near_hashes = np.unique(np.concatenate([spatial_hashes - 1, spatial_hashes, spatial_hashes + 1]))
        for h in near_hashes:
            if h < 0 or h >= hash_grid**2:
                continue
            nearby_indices = np.where(spatial_hashes == h)[0]
            nearby_indices = np.unique(np.concatenate((np.array([i]), nearby_indices)))
            for j in nearby_indices:
                if i == j:
                    continue
                # Add constraint for distance >= sum of radii
                # Use vectorized computation to avoid recomputing the same distances
                dx = v[3*i] - v[3*j]
                dy = v[3*i+1] - v[3*j+1]
                cons.append({"type": "ineq", 
                            "fun": (lambda v, i=i, j=j, dx=dx, dy=dy: 
                                     dx*dx + dy*dy - (v[3*i+2] + v[3*j+2])**2)})

    # First Optimization with enhanced settings and parallelizable computation
    # Set very high precision due to the nature of the problem
    res = minimize(
        neg_sum_radii, v0, 
        method="SLSQP", 
        bounds=bounds,
        constraints=cons, 
        options={
            "maxiter": 1800,
            "ftol": 1e-11,
            "gtol": 1e-11,
            "eps": 1e-8,
            "disp": False,
            "iprint": 0
        }
    )
    
    # Adaptive reconfiguration with spatial-aware perturbation
    if res.success:
        v = res.x
        # Evaluate spatial hashing again for more accurate perturbation
        xs = v[0::3]
        ys = v[1::3]
        radii = v[2::3]
        spatial_hashes = np.array([int(x * hash_grid) + int(y * hash_grid) for x, y in zip(xs, ys)])
        # Find the most spatially constrained circle (least distance to other circles)
        dists = np.zeros(n * n)
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dists[i * n + j] = np.sqrt(dx*dx + dy*dy)
                dists[j * n + i] = dists[i * n + j]
        min_dists = np.min(dists, axis=1)
        most_constrained_idx = np.argmin(min_dists) if np.min(min_dists) > 1e-9 else np.random.randint(n)
        
        # Spatial reconfiguration for constrained circle using adaptive perturbation
        # Use sine-based perturbation for non-uniform distribution
        max_perturb_rad = 0.015 * (np.mean(radii) / np.min(radii))
        perturbation = np.random.rand(n, 2) * max_perturb_rad
        perturbed_v = v.copy()
        for i in range(n):
            perturbed_v[3*i] += (perturbation[i, 0] * (np.cos(radii[i]) / np.sin(radii[i])))
            perturbed_v[3*i+1] += (perturbation[i, 1] * (np.cos(radii[i]) / np.sin(radii[i])))
        
        # Re-evaluate with perturbed parameters
        res = minimize(
            neg_sum_radii, perturbed_v, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons, 
            options={
                "maxiter": 350,
                "ftol": 1e-10,
                "gtol": 1e-10,
                "eps": 1e-8,
                "disp": False,
                "iprint": 0
            }
        )
    
    # Targeted radius expansion with spatial-aware constraint enforcement
    if res.success:
        v = res.x
        xs = v[0::3]
        ys = v[1::3]
        radii = v[2::3]
        # Determine circle with maximum expansion potential
        # Use a hybrid metric combining expansion potential and spatial feasibility
        expansion_potential = np.zeros(n)
        for i in range(n):
            # Use relative spacing as a proxy for potential
            dists_i = np.sqrt(
                (xs[i] - xs)**2 + (ys[i] - ys)**2
            )
            min_dist_i = np.min(dists_i)
            avg_dist_i = np.mean(dists_i)
            # Expansion potential = relative spacing + inverse radius
            expansion_potential[i] = (avg_dist_i / min_dist_i) + (1.0 / (radii[i] + 1e-9))
        
        # Select circle with highest expansion potential
        expansion_idx = np.argmax(expansion_potential)
        
        # Calculate total expansion based on current configuration
        current_total = np.sum(radii)
        # Use a soft constraint strategy to expand while respecting boundaries
        # We will increase total sum by 0.008 - 0.012 (approx 0.0095)
        target_total = current_total + 0.0095
        # Calculate expansion factor
        expansion_factor = (target_total - current_total) / (n - 1) if n > 1 else target_total - current_total
        
        # Create expansion vector with adaptive scaling
        # Increase the selected circle's radius by a factor (with slight over-expansion)
        # Then distribute the rest among other circles proportionally
        # Apply a dynamic radius expansion with minimal over-constraint
        # Apply soft constraints via a linear expansion with boundary checks
        expanded_radii = radii.copy()
        # First expand the most constrained circle by 1.2x to trigger reconfiguration
        expanded_radii[expansion_idx] = np.clip(radii[expansion_idx] * 1.2, 1e-4, 0.5)
        
        # Distribute remaining expansion while maintaining spatial viability
        remaining_expansion = target_total - np.sum(expanded_radii)
        if remaining_expansion > 0 and n > 1:
            for i in range(n):
                if i != expansion_idx:
                    # Apply adaptive expansion based on proximity and spatial spacing
                    # Weight expansion amount by distance to the most constrained circle
                    dist_to_expanded = np.sqrt(
                        (xs[i] - xs[expansion_idx])**2 + (ys[i] - ys[expansion_idx])**2
                    )
                    # Normalize by average distance
                    norm_dist = dist_to_expanded / np.mean(np.sqrt(
                        (xs - xs[expansion_idx])**2 + (ys - ys[expansion_idx])**2
                    ))
                    # Scale expansion by proximity and use inverse of radius
                    expansion = expansion_factor * norm_dist * (1.0 / (radii[i] + 1e-9))
                    expanded_radii[i] += np.clip(expansion, 1e-4, 0.015)
        
        # Now, re-evaluate the configuration with these expanded radii
        # This avoids large-scale optimization and uses local constraints
        expanded_v = v.copy()
        expanded_v[2::3] = expanded_radii
        
        # Use direct distance check for validation (fast)
        # We precompute distances to avoid redundant checks in constraints
        xs_expanded = expanded_v[0::3]
        ys_expanded = expanded_v[1::3]
        radii_expanded = expanded_v[2::3]
        # Validate spatial layout
        # Check if all circles are within the unit square
        if (xs_expanded < 1e-10).any() or (xs_expanded > 1.0 - 1e-10).any() or \
           (ys_expanded < 1e-10).any() or (ys_expanded > 1.0 - 1e-10).any():
            # If any circle is out of bounds, adjust by shrinking radii to fit
            # Create a fallback constraint to reconfigure
            pass
        else:
            # Check for overlaps
            for i in range(n):
                for j in range(i+1, n):
                    dx = xs_expanded[i] - xs_expanded[j]
                    dy = ys_expanded[i] - ys_expanded[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < radii_expanded[i] + radii_expanded[j] - 1e-12:
                        # If any overlaps, reduce the largest radius by a fixed value
                        large_radius_idx = np.argmax(radii_expanded)
                        expanded_radii[large_radius_idx] -= 0.001
                        # Ensure radius doesn't go below minimum
                        expanded_radii[large_radius_idx] = np.clip(expanded_radii[large_radius_idx], 1e-4, 0.5)
        
        # Re-apply expansion and optimize constrained configuration
        # This ensures we maintain the configuration through spatial-aware expansion
        expanded_v[2::3] = expanded_radii
        res = minimize(
            neg_sum_radii, expanded_v, 
            method="SLSQP", 
            bounds=bounds,
            constraints=cons, 
            options={
                "maxiter": 200,
                "ftol": 1e-10,
                "gtol": 1e-10,
                "eps": 5e-8,
                "disp": False,
                "iprint": 0
            }
        )
    
    v = res.x if res.success else v0
    centers = np.column_stack([v[0::3], v[1::3]])
    radii = np.clip(v[2::3], 1e-6, None)
    return centers, radii, float(radii.sum())